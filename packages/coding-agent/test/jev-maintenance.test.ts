import { afterEach, describe, expect, it, vi } from "bun:test";
import * as path from "node:path";
import { Agent, type AgentMessage } from "@oh-my-pi/pi-agent-core";
import * as compactionModule from "@oh-my-pi/pi-agent-core/compaction";
import {
	type JudgmentRequest,
	type JudgmentResult,
	type Questions,
	type ToolResultMessage,
	TypeSafeJudge,
} from "@oh-my-pi/pi-ai";
import { getBundledModel } from "@oh-my-pi/pi-catalog/models";
import { ModelRegistry } from "@oh-my-pi/pi-coding-agent/config/model-registry";
import { Settings } from "@oh-my-pi/pi-coding-agent/config/settings";
import { AgentSession } from "@oh-my-pi/pi-coding-agent/session/agent-session";
import { AuthStorage } from "@oh-my-pi/pi-coding-agent/session/auth-storage";
import * as artifactModule from "@oh-my-pi/pi-coding-agent/session/artifacts";
import { SessionManager } from "@oh-my-pi/pi-coding-agent/session/session-manager";
import { TempDir } from "@oh-my-pi/pi-utils";

function usage(input: number) {
	return {
		input,
		output: 1,
		cacheRead: 0,
		cacheWrite: 0,
		totalTokens: input + 1,
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
	};
}

function answer<Q extends Questions>(request: JudgmentRequest<Q>): JudgmentResult<Q> {
	return {
		api: "typesafe",
		provider: "typesafe",
		model: "jev-latest",
		answers: Object.fromEntries(
			Object.keys(request.questions).map(id => [id, { type: "noul", noul: 0.99 }]),
		) as JudgmentResult<Q>["answers"],
		usage: usage(17),
	};
}

function results(messages: readonly AgentMessage[]) {
	return messages.filter(message => message.role === "toolResult");
}

function resultText(messages: readonly AgentMessage[], id: string): string {
	const message = results(messages).find(message => message.toolCallId === id);
	if (!message) throw new Error(`Missing tool result ${id}`);
	return message.content.flatMap(block => (block.type === "text" ? [block.text] : [])).join("\n");
}

// Spies are scoped to each test and restored after session disposal; no module mocks.
describe("native Jev maintenance transactions", () => {
	let temp: TempDir;
	let auth: AuthStorage;
	let manager: SessionManager;
	let session: AgentSession;
	let original: AgentMessage[];
	let bodies: string[];
	const releases: Array<() => void> = [];

	async function setup(providerTokens = 40_000, fallback = false) {
		temp = TempDir.createSync("@jev-maintenance-");
		auth = await AuthStorage.create(path.join(temp.path(), "auth.db"));
		auth.setRuntimeApiKey("anthropic", "test-only");
		auth.setRuntimeApiKey("typesafe", "test-only");
		manager = SessionManager.create(temp.path(), temp.path());
		const model = getBundledModel("anthropic", "claude-sonnet-4-5");
		if (!model) throw new Error("Missing bundled test model");
		const agent = new Agent({ initialState: { model, systemPrompt: ["Test"], tools: [], messages: [] } });
		const user = (text: string) => manager.appendMessage({ role: "user", content: text, timestamp: Date.now() });
		const assistant = (content: Extract<AgentMessage, { role: "assistant" }>["content"], input = 0) =>
			manager.appendMessage({
				role: "assistant",
				content,
				api: "anthropic-messages",
				provider: "anthropic",
				model: model.id,
				stopReason: "stop",
				usage: usage(input),
				timestamp: Date.now(),
			});
		const pair = (id: string, name: string, text: string, isError = false) => {
			assistant([{ type: "toolCall", id, name, arguments: { path: `${id}.txt` } }]);
			manager.appendMessage({
				role: "toolResult",
				toolCallId: id,
				toolName: name,
				content: [{ type: "text", text }],
				isError,
				timestamp: Date.now(),
			});
		};
		const tokenBody = (prefix: string, tokens: number) => {
			let text = prefix;
			while (agent.tokenizer.countTokens(text) < tokens) text += "a1 b2 c3 d4 ".repeat(100);
			return text;
		};
		bodies = [tokenBody("older alpha\n", 6_000), tokenBody("older beta\n", 6_000)];
		user("Inspect old files; retain all inputs and receipts.");
		pair("old-a", "read", bodies[0]);
		pair("old-b", "read", bodies[1]);
		pair("failed", "read", "ENOENT: required file unavailable", true);
		pair("shell", "bash", "mutation command completed");
		pair("written", "write", "saved durable configuration");
		assistant([{ type: "text", text: "The receipts above describe completed work." }]);
		user("Recent first turn");
		pair("tail", "read", tokenBody("protected live tail\n", 18_000));
		user("Recent second turn");
		pair("recent", "read", "keep this recent finding");
		user("Recent third turn: continue inspecting, do not repeat writes.");
		assistant([{ type: "text", text: "Ready." }], providerTokens);
		await manager.ensureOnDisk();
		original = structuredClone(manager.buildSessionContext().messages);
		agent.replaceMessages(manager.buildSessionContext().messages);
		session = new AgentSession({
			agent,
			sessionManager: manager,
			modelRegistry: new ModelRegistry(auth),
			settings: Settings.isolated({
				"compaction.methodOrder": fallback ? ["jev", "soft"] : ["jev"],
				"compaction.enabled": true,
				"compaction.autoContinue": false,
				"compaction.thresholdTokens": 40_000,
				"compaction.keepRecentTokens": 1,
				"contextPromotion.enabled": false,
				"todo.enabled": false,
				"todo.reminders": false,
			}),
		});
		vi.spyOn(TypeSafeJudge.prototype, "judge").mockImplementation(async request => answer(request));
	}

	afterEach(async () => {
		for (const release of releases.splice(0)) release();
		await session?.dispose();
		auth?.close();
		vi.restoreAllMocks();
		await temp?.remove();
	});

	it("persists recoverable elisions across reopen without changing calls, receipts or recent turns", async () => {
		await setup();
		await session.runIdleCompaction();
		const current = manager.buildSessionContext().messages;
		for (const id of ["failed", "shell", "written", "tail", "recent"]) {
			expect(resultText(current, id)).toBe(resultText(original, id));
		}
		// Usage anchors may be adjusted, but assistant inputs/prose and every user turn are immutable.
		const inputs = (messages: readonly AgentMessage[]) =>
			messages
				.filter(message => message.role === "user" || message.role === "assistant")
				.map(message => ({ role: message.role, content: message.content }));
		expect(inputs(current)).toEqual(inputs(original));
		const artifactIds = ["old-a", "old-b"].map(id => {
			const match = /artifact:\/\/(\d+)/.exec(resultText(current, id));
			if (!match) throw new Error(`No recovery pointer for ${id}`);
			return match[1];
		});
		const artifactPath = await manager.getArtifactPath(artifactIds[0]);
		if (!artifactPath) throw new Error("Recovery artifact is not durable");
		const bytes = await Bun.file(artifactPath).bytes();
		const artifact = new TextDecoder().decode(bytes);
		for (const body of bodies) expect(artifact).toContain(body);
		expect(artifactIds[1]).toBe(artifactIds[0]);
		const ledger = manager.getEntries().filter(entry => entry.type === "model_usage");
		expect(ledger.some(entry => entry.purpose === "jev-compaction" && entry.usage.input === 17)).toBe(true);
		expect(current.map(message => message.role)).toEqual(original.map(message => message.role));
		await manager.flush();
		const file = manager.getSessionFile();
		if (!file) throw new Error("Missing session file");
		const reopened = await SessionManager.open(file, temp.path());
		try {
			const restored = reopened.buildSessionContext().messages;
			expect(inputs(restored)).toEqual(inputs(current));
			for (const result of results(current)) {
				expect(resultText(restored, result.toolCallId)).toBe(resultText(current, result.toolCallId));
			}
			const recoveredPath = await reopened.getArtifactPath(artifactIds[0]);
			expect(await Bun.file(recoveredPath!).bytes()).toEqual(bytes);
			expect(reopened.getEntries().filter(entry => entry.type === "model_usage")).toEqual(ledger);
		} finally {
			await reopened.close();
		}
	});

	it("rejects inadequate provider-anchored savings atomically and gives fallback the original results", async () => {
		await setup(100_000, true);
		let fallbackResults: ToolResultMessage[] | undefined;
		vi.spyOn(compactionModule, "compact").mockImplementation(async preparation => {
			fallbackResults = structuredClone(
				results([
					...preparation.messagesToSummarize,
					...preparation.turnPrefixMessages,
					...preparation.recentMessages,
				]),
			);
			throw new Error("Stop before calling a live summary provider");
		});
		await session.runIdleCompaction();
		expect(fallbackResults).toEqual(results(original));
		expect(results(manager.buildSessionContext().messages)).toEqual(results(original));
	});

	for (const failure of ["allocation", "write"] as const) {
		it(`leaves all original results intact on recovery artifact ${failure} failure`, async () => {
			await setup();
			const allocation = vi.spyOn(manager, "allocateArtifactPath");
			const artifactWrite = vi.spyOn(artifactModule, "writeArtifact");
			if (failure === "allocation") {
				allocation.mockRejectedValue(new Error("disk unavailable"));
			} else {
				// A directory is not a writable artifact file, independently of uid/permissions.
				allocation.mockResolvedValue({ id: "999", path: temp.path() });
			}
			await session.runIdleCompaction();
			expect(allocation).toHaveBeenCalledTimes(1);
			expect(
				manager.getEntries().some(entry => entry.type === "model_usage" && entry.purpose === "jev-compaction"),
			).toBe(true);
			if (failure === "write") expect(artifactWrite).toHaveBeenCalledTimes(1);
			else expect(artifactWrite).not.toHaveBeenCalled();
			expect(results(manager.buildSessionContext().messages)).toEqual(results(original));
			await manager.flush();
			const reopened = await SessionManager.open(manager.getSessionFile()!, temp.path());
			try {
				expect(results(reopened.buildSessionContext().messages)).toEqual(results(original));
			} finally {
				await reopened.close();
			}
		});
	}

	for (const interruption of ["branch", "abort"] as const) {
		it(`does not commit or run fallback after ${interruption} while the judge is pending`, async () => {
			await setup(40_000, true);
			const entered = Promise.withResolvers<void>();
			const release = Promise.withResolvers<void>();
			releases.push(() => release.resolve());
			vi.spyOn(TypeSafeJudge.prototype, "judge").mockImplementation(async request => {
				entered.resolve();
				await release.promise;
				return answer(request);
			});
			const fallback = vi.spyOn(compactionModule, "compact").mockRejectedValue(new Error("Unexpected fallback"));
			const pending = session.runIdleCompaction();
			await Promise.race([
				entered.promise,
				pending.then(() => {
					throw new Error("Jev never reached judgment");
				}),
			]);
			let branchLeaf: string | null = null;
			if (interruption === "branch") {
				const root = manager.getBranch().find(entry => entry.type === "message");
				if (!root) throw new Error("Missing branch root");
				manager.branch(root.id);
				branchLeaf = manager.appendMessage({
					role: "user",
					content: "New branch must remain untouched",
					timestamp: Date.now(),
				});
			} else {
				session.abortCompaction();
			}
			release.resolve();
			await pending;
			expect(fallback).not.toHaveBeenCalled();
			expect(
				results(manager.getEntries().flatMap(entry => (entry.type === "message" ? [entry.message] : []))),
			).toEqual(results(original));
			if (branchLeaf) expect(manager.getLeafId()).toBe(branchLeaf);
			expect(manager.getEntries().some(entry => entry.type === "compaction")).toBe(false);
		});
	}
});
