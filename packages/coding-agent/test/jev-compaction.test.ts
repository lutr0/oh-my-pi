import { afterEach, describe, expect, it, vi } from "bun:test";
import type { SessionEntry, SessionMessageEntry, ShakeRegion } from "@oh-my-pi/pi-agent-core/compaction";
import type { Judge, JudgmentRequest, JudgmentResult, Questions } from "@oh-my-pi/pi-ai";
import { selectJevShakeRegions } from "@oh-my-pi/pi-coding-agent/session/jev-compaction";

function entry(id: string, message: unknown): SessionMessageEntry {
	return { type: "message", id, parentId: null, timestamp: "2026-01-01T00:00:00Z", message } as SessionMessageEntry;
}

function pair(id: string, body = "old output", name = "read") {
	const call = entry(`call-${id}`, {
		role: "assistant",
		content: [{ type: "toolCall", id, name, arguments: { path: "secret/path" } }],
	});
	const result = entry(`result-${id}`, {
		role: "toolResult",
		toolCallId: id,
		toolName: name,
		content: [{ type: "text", text: body }],
		isError: false,
	});
	const region: ShakeRegion = { kind: "toolResult", entry: result, originalText: body, label: name, tokens: 999_999 };
	return { call, result, region };
}

function tail(): SessionEntry[] {
	return [
		entry("u1", { role: "user", content: "secret active task" }),
		entry("a1", {
			role: "assistant",
			content: [
				{ type: "thinking", thinking: "private reasoning" },
				{ type: "text", text: "secret plan" },
			],
		}),
		entry("u2", { role: "user", content: "continue" }),
		entry("a2", { role: "assistant", content: [{ type: "text", text: "Continuing." }] }),
		entry("u3", { role: "user", content: "next step" }),
		entry("a3", { role: "assistant", content: [{ type: "text", text: "Checking the next step." }] }),
	];
}

function response(request: JudgmentRequest, value = 0.95): JudgmentResult {
	return {
		api: "typesafe",
		provider: "typesafe",
		model: "jev-latest",
		answers: Object.fromEntries(Object.keys(request.questions).map(id => [id, { type: "noul", noul: value }])),
		usage: {
			input: 1,
			output: 1,
			cacheRead: 0,
			cacheWrite: 0,
			totalTokens: 2,
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
		},
	};
}

function judgeWith(
	run: (request: JudgmentRequest, signal?: AbortSignal) => Promise<JudgmentResult> | JudgmentResult,
): Judge {
	return {
		label: "typesafe/jev-latest",
		judge: ((request: JudgmentRequest, options?: { signal?: AbortSignal }) =>
			Promise.resolve(run(request, options?.signal))) as Judge["judge"],
	};
}

const obfuscate = (text: string) => text;
afterEach(() => vi.restoreAllMocks());

describe("Jev recoverable selection", () => {
	it("offloads old results during a long tool loop with only one human prompt", async () => {
		const old = pair("old");
		const recent = [pair("recent-a"), pair("recent-b"), pair("recent-c")];
		const entries = [
			entry("task", { role: "user", content: "Continue the long-running task." }),
			old.call,
			old.result,
			...recent.flatMap(p => [p.call, p.result]),
		];
		expect(
			await selectJevShakeRegions({
				entries,
				regions: [old.region, ...recent.map(p => p.region)],
				judge: judgeWith(request => response(request)),
				obfuscate,
			}),
		).toEqual([old.region]);
	});

	it("pins unsafe pairs, recent calls/results, prose and caller-protected results", async () => {
		const safe = pair("safe");
		const error = pair("error");
		if (error.result.message.role === "toolResult") error.result.message.isError = true;
		const mixed = pair("mixed");
		if (mixed.result.message.role === "toolResult")
			mixed.result.message.content.push({ type: "image", data: "image", mimeType: "image/png" });
		const pruned = pair("pruned");
		if (pruned.result.message.role === "toolResult") pruned.result.message.prunedAt = 1;
		const duplicate = pair("duplicate");
		const duplicateResult = pair("duplicate-result");
		const mismatch = pair("mismatch");
		if (mismatch.result.message.role === "toolResult") mismatch.result.message.toolName = "glob";
		const unknown = pair("unknown", "output", "bash");
		const orphan = pair("orphan");
		const pending = pair("pending");
		const protectedPair = pair("protected");
		const crossing = pair("crossing");
		const recent = pair("recent");
		const reversed = pair("reversed");
		const old = [safe, error, mixed, pruned, duplicate, duplicateResult, mismatch, unknown, protectedPair];
		const live = tail();
		const entries = [
			...old.flatMap(p => [p.call, p.result]),
			duplicate.call,
			duplicateResult.result,
			orphan.result,
			pending.call,
			reversed.result,
			reversed.call,
			crossing.call,
			live[0],
			recent.call,
			crossing.result,
			recent.result,
			...live.slice(2),
		];
		const regions = [
			...old.filter(p => p !== protectedPair).map(p => p.region),
			orphan.region,
			pending.region,
			reversed.region,
			crossing.region,
			recent.region,
			{
				kind: "block",
				entry: live[0],
				blockIndex: -1,
				start: 0,
				end: 5,
				tokens: 10,
				originalText: "prose",
				label: "user",
			} as ShakeRegion,
		];
		const before = JSON.stringify({ entries, regions });
		const requests: JudgmentRequest[] = [];
		const judge = judgeWith(request => {
			requests.push(request);
			return response(request);
		});
		expect(await selectJevShakeRegions({ entries, regions, judge, obfuscate })).toEqual([safe.region]);
		expect(JSON.stringify(requests)).not.toContain("protected");
		expect(JSON.stringify({ entries, regions })).toBe(before);
		expect(
			await selectJevShakeRegions({
				entries: [safe.call, safe.result, ...tail().slice(2)],
				regions: [safe.region],
				judge,
				obfuscate,
			}),
		).toEqual([]);
		expect(requests).toHaveLength(1);
	});

	it("scores complete bodies with explicit ids and obfuscates context, arguments and body", async () => {
		const p = pair("visible", `secret start${"x".repeat(8000)}secret end`);
		const judge = judgeWith(request => {
			const serialized = String(request.state);
			expect(serialized).not.toContain("secret");
			expect(serialized).not.toContain("private reasoning");
			const state = JSON.parse(serialized);
			expect(state.results[0].body).toEqual([p.region.originalText.replaceAll("secret", "REDACTED")]);
			expect(state.results[0].arguments.path).toBe("REDACTED/path");
			expect(state.context[0].text).toEqual(["REDACTED active task"]);
			expect(state.context[1].text).toEqual(["REDACTED plan"]);
			expect(request.questions[state.results[0].id].instructions).toContain(state.results[0].id);
			return response(request, 0.9);
		});
		expect(
			await selectJevShakeRegions({
				entries: [p.call, p.result, ...tail()],
				regions: [p.region],
				judge,
				obfuscate: text => text.replaceAll("secret", "REDACTED"),
			}),
		).toEqual([p.region]);
	});

	it("pins oversized bodies and context and caps sequential requests at eight within UTF8 budgets", async () => {
		const huge = pair("huge", "界".repeat(12_000));
		const pairs = Array.from({ length: 10 }, (_, i) => pair(String(i), "界".repeat(6500)));
		let calls = 0;
		let active = false;
		const judge = judgeWith(async request => {
			expect(active).toBe(false);
			active = true;
			calls++;
			const longest = Math.max(...Object.values(request.questions).map(q => Buffer.byteLength(JSON.stringify(q))));
			expect(Buffer.byteLength(JSON.stringify(request.state)) + longest).toBeLessThanOrEqual(28_000);
			expect(Buffer.byteLength(JSON.stringify(request))).toBeLessThanOrEqual(56_000);
			expect(String(request.state)).not.toContain('result_1"');
			await Promise.resolve();
			active = false;
			return response(request);
		});
		const entries = [huge.call, huge.result, ...pairs.flatMap(p => [p.call, p.result]), ...tail()];
		const regions = [huge.region, ...pairs.map(p => p.region)];
		expect(await selectJevShakeRegions({ entries, regions, judge, obfuscate })).toEqual(
			pairs.slice(0, 8).map(p => p.region),
		);
		expect(calls).toBe(8);
		expect(
			await selectJevShakeRegions({
				entries: [
					...entries,
					entry("large-context", { role: "assistant", content: [{ type: "text", text: "界".repeat(12_000) }] }),
				],
				regions,
				judge,
				obfuscate,
			}),
		).toEqual([]);
		expect(calls).toBe(8);
		expect(
			await selectJevShakeRegions({ entries, regions, judge, obfuscate: text => `${text}${"x".repeat(30_000)}` }),
		).toEqual([]);
	});

	it("rejects invalid later responses atomically while reporting every received usage", async () => {
		const pairs = [pair("one", "x".repeat(20_000)), pair("two", "x".repeat(20_000))];
		for (const invalid of [
			undefined,
			{ type: "noul", noul: NaN },
			{ type: "noul", noul: Infinity },
			{ type: "noul", noul: -0.1 },
			{ type: "noul", noul: 1.1 },
			{ type: "noul", noul: "0.99" },
			{ type: "score", score: 1 },
		]) {
			let calls = 0;
			const usages: JudgmentResult<Questions>[] = [];
			const judge = judgeWith(request => {
				const result = response(request);
				if (++calls === 2)
					result.answers = { [Object.keys(request.questions)[0]]: invalid } as JudgmentResult["answers"];
				return result;
			});
			await expect(
				selectJevShakeRegions({
					entries: [...pairs.flatMap(p => [p.call, p.result]), ...tail()],
					regions: pairs.map(p => p.region),
					judge,
					obfuscate,
					onUsage: result => usages.push(result),
				}),
			).rejects.toThrow("Invalid Jev answer");
			expect(usages).toHaveLength(2);
		}
	});

	it("retains uncertain bodies below the conservative threshold", async () => {
		const p = pair("uncertain");
		expect(
			await selectJevShakeRegions({
				entries: [p.call, p.result, ...tail()],
				regions: [p.region],
				judge: judgeWith(request => response(request, 0.899)),
				obfuscate,
			}),
		).toEqual([]);
	});

	it("propagates caller abort even when the backend ignores cancellation", async () => {
		const p = pair("abort");
		const controller = new AbortController();
		const reason = new Error("caller cancelled");
		let backendSignal: AbortSignal | undefined;
		const judge = judgeWith((_request, signal) => {
			backendSignal = signal;
			queueMicrotask(() => controller.abort(reason));
			return Promise.withResolvers<JudgmentResult>().promise;
		});
		await expect(
			selectJevShakeRegions({
				entries: [p.call, p.result, ...tail()],
				regions: [p.region],
				judge,
				obfuscate,
				signal: controller.signal,
			}),
		).rejects.toBe(reason);
		expect(backendSignal?.aborted).toBe(true);
	});

	it("uses one overall deadline across batches and cancels an unresponsive backend", async () => {
		const pairs = [pair("one", "x".repeat(20_000)), pair("two", "x".repeat(20_000))];
		let fireDeadline: (() => void) | undefined;
		const realSetTimeout = globalThis.setTimeout;
		const timers = vi.spyOn(globalThis, "setTimeout").mockImplementation(((callback: () => void, delay?: number) => {
			expect(delay).toBe(15_000);
			fireDeadline = callback;
			return realSetTimeout(callback, 60_000);
		}) as typeof setTimeout);
		const clear = vi.spyOn(globalThis, "clearTimeout");
		let calls = 0;
		let backendSignal: AbortSignal | undefined;
		const judge = judgeWith((request, signal) => {
			backendSignal = signal;
			if (++calls === 1) return response(request);
			queueMicrotask(() => fireDeadline?.());
			return Promise.withResolvers<JudgmentResult>().promise;
		});
		await expect(
			selectJevShakeRegions({
				entries: [...pairs.flatMap(p => [p.call, p.result]), ...tail()],
				regions: pairs.map(p => p.region),
				judge,
				obfuscate,
			}),
		).rejects.toThrow("deadline exceeded");
		expect(calls).toBe(2);
		expect(timers).toHaveBeenCalledTimes(1);
		expect(clear).toHaveBeenCalledTimes(1);
		expect(backendSignal?.aborted).toBe(true);
	});
});
