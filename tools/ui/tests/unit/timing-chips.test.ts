import { afterEach, beforeAll, describe, expect, it, vi } from 'vitest';

beforeAll(() => {
	const store = new Map<string, string>();

	(globalThis as unknown as { localStorage: Storage }).localStorage = {
		clear: () => store.clear(),
		getItem: (k: string) => store.get(k) ?? null,
		key: () => null,
		get length() {
			return store.size;
		},
		removeItem: (k: string) => void store.delete(k),
		setItem: (k: string, v: string) => void store.set(k, String(v))
	} as Storage;
});

import { MessageRole, ToolPermissionDecision, ToolSource } from '$lib/enums';
import { ChatService } from '$lib/services/chat.service';
import { agenticStore } from '$lib/stores/agentic/index.svelte';
import { mcpStore } from '$lib/stores/mcp/index.svelte';
import { modelsStore } from '$lib/stores/models/index.svelte';
import { toolsStore } from '$lib/stores/tools.svelte';
import type { ChatMessageTimings, DatabaseMessage } from '$lib/types';
import {
	cumulativeUsage,
	formatClock,
	sessionLine,
	timingChips,
	toolChip,
	toolRecordIndex,
	turnSummary
} from '$lib/utils/timing-chips';

const msg = (
	role: MessageRole,
	timings?: ChatMessageTimings,
	extra: Partial<DatabaseMessage> = {}
) =>
	({
		children: [],
		content: 'x',
		convId: 'c',
		id: Math.random().toString(36),
		parent: null,
		role,
		timestamp: 1000,
		toolCalls: '',
		type: 'text',
		...(timings ? { timings } : {}),
		...extra
	}) as DatabaseMessage;

function sse(chunks: object[]): Response {
	const body = chunks.map((c) => `data: ${JSON.stringify(c)}\n\n`).join('') + 'data: [DONE]\n\n';

	return new Response(body, { headers: { 'Content-Type': 'text/event-stream' } });
}

afterEach(() => {
	vi.restoreAllMocks();
	vi.unstubAllGlobals();
});

describe('W2 timing chips', () => {
	it('plain reply: total, TTFT, prefill fresh + cached with cache share and t/s, decode t/s, MTP %', () => {
		const c = timingChips({
			cache_n: 9800,
			client: { t_start: 0, ttft_ms: 710, wall_ms: 10230 },
			draft_n: 721,
			draft_n_accepted: 512,
			predicted_ms: 9370,
			predicted_n: 386,
			prompt_ms: 650,
			prompt_n: 1204
		})!;

		expect(c.totalMs).toBe(10230);
		expect(c.ttftMs).toBe(710);
		expect(c.prefill).toEqual({
			cached: 9800,
			cacheShare: 9800 / 11004,
			fresh: 1204,
			ms: 650,
			tps: (1204 / 650) * 1000
		});
		expect(c.decode).toEqual({ ms: 9370, n: 386, tps: (386 / 9370) * 1000 });
		expect(c.mtp).toBeCloseTo(512 / 721);
	});

	it('agentic root: the flow totals and the flow wall', () => {
		const c = timingChips({
			agentic: {
				llm: {
					cache_n: 5000,
					draft_n: 100,
					draft_n_accepted: 60,
					predicted_ms: 2000,
					predicted_n: 100,
					prompt_ms: 1000,
					prompt_n: 500
				},
				t_end: 9000,
				t_start: 1000,
				toolCallsCount: 2,
				toolsMs: 3000,
				turns: 3
			},
			predicted_n: 10
		})!;

		expect(c.totalMs).toBe(8000);
		expect(c.prefill?.cacheShare).toBeCloseTo(5000 / 5500);
		expect(c.decode?.n).toBe(100);
		expect(c.mtp).toBeCloseTo(0.6);
	});

	it('legacy timings without client / draft fields still chip; no timings = none', () => {
		expect(timingChips({ predicted_ms: 100, predicted_n: 10, prompt_ms: 0, prompt_n: 0 })).toEqual({
			decode: { ms: 100, n: 10, tps: 100 },
			totalMs: undefined,
			ttftMs: undefined
		});
		expect(timingChips(undefined)).toBeNull();
	});
});

describe('W2 turn summary line', () => {
	it('wall = model (prefill + decode) + tools + other', () => {
		const t = turnSummary({
			agentic: {
				llm: { predicted_ms: 37800, predicted_n: 1, prompt_ms: 3200, prompt_n: 1 },
				t_end: 96000,
				t_start: 0,
				toolCallsCount: 9,
				toolsMs: 49800,
				turns: 6
			}
		})!;

		expect(t).toEqual({
			decodeMs: 37800,
			llmCalls: 6,
			modelMs: 41000,
			otherMs: 96000 - 41000 - 49800,
			prefillMs: 3200,
			toolCalls: 9,
			toolsMs: 49800,
			wallMs: 96000
		});
		expect(t.modelMs + t.toolsMs + t.otherMs).toBe(t.wallMs);
	});

	it('plain reply: the request wall; fallback to message timestamps; other never < 0', () => {
		expect(
			turnSummary({ client: { t_start: 0, wall_ms: 1000 }, predicted_ms: 700, prompt_ms: 200 })
		).toMatchObject({ llmCalls: 1, modelMs: 900, otherMs: 100, toolsMs: 0, wallMs: 1000 });
		expect(
			turnSummary({ predicted_ms: 700, prompt_ms: 400 }, { endMs: 2000, startMs: 1000 })
		).toMatchObject({ otherMs: 0, wallMs: 1000 });
		expect(turnSummary({ predicted_ms: 700, prompt_ms: 400 })).toBeNull();
	});
});

describe('W2 tool chips', () => {
	it('binds records by tool-call id; running clock; legacy sizes from the stored args / result', () => {
		const rec = {
			args_bytes: 86,
			duration_ms: 12400,
			name: 'exec',
			result_bytes: 18200,
			success: true,
			toolCallId: 'call_1'
		};
		const idx = toolRecordIndex([
			msg(MessageRole.ASSISTANT, {
				agentic: {
					llm: { predicted_ms: 0, predicted_n: 0, prompt_ms: 0, prompt_n: 0 },
					toolCalls: [rec, { duration_ms: 1, name: 'old', success: true }],
					toolCallsCount: 2,
					toolsMs: 0,
					turns: 1
				}
			})
		]);

		expect(idx.get('call_1')).toBe(rec);
		expect(idx.size).toBe(1);
		expect(toolChip({ record: rec })).toEqual({
			argsBytes: 86,
			durationMs: 12400,
			error: undefined,
			ok: true,
			resultBytes: 18200,
			running: false
		});
		expect(toolChip({ args: '{"a":1}', executingSince: 1000, now: 5100 })).toEqual({
			argsBytes: 7,
			durationMs: 4100,
			running: true
		});
		expect(toolChip({ args: 'é', result: 'ok' })).toMatchObject({ argsBytes: 2, resultBytes: 2 });
	});

	it('the agentic loop records id, start / end, sizes and error per call; D3 keeps draft + cache', async () => {
		vi.spyOn(toolsStore, 'getToolSource').mockReturnValue(ToolSource.MCP);
		vi.spyOn(toolsStore, 'getToolServerLabel').mockReturnValue('srv');
		vi.spyOn(modelsStore.props, 'modelSupportsVision').mockReturnValue(false);
		vi.spyOn(
			(agenticStore as unknown as { gates: { requestPermission: () => Promise<string> } }).gates,
			'requestPermission'
		).mockResolvedValue(ToolPermissionDecision.ONCE);
		vi.spyOn(mcpStore, 'executeTool').mockRejectedValue(new Error('boom'));

		let n = 0;

		vi.stubGlobal(
			'fetch',
			vi.fn(async () =>
				n++ === 0
					? sse([
							{
								choices: [
									{
										delta: {
											tool_calls: [
												{
													function: { arguments: '{"q":"é"}', name: 'shot' },
													id: 'call_9',
													index: 0,
													type: 'function'
												}
											]
										}
									}
								],
								timings: {
									cache_n: 900,
									draft_n: 10,
									draft_n_accepted: 7,
									predicted_ms: 50,
									predicted_n: 5,
									prompt_ms: 20,
									prompt_n: 100
								}
							}
						])
					: sse([
							{
								choices: [{ delta: { content: 'done' }, finish_reason: 'stop' }],
								timings: {
									cache_n: 1000,
									draft_n: 20,
									draft_n_accepted: 15,
									predicted_ms: 40,
									predicted_n: 4,
									prompt_ms: 10,
									prompt_n: 30
								}
							}
						])
			)
		);

		let final: ChatMessageTimings | undefined;

		await (
			agenticStore as unknown as { executeAgenticLoop: (p: unknown) => Promise<void> }
		).executeAgenticLoop({
			agenticConfig: { enabled: true, maxTurns: 5 },
			callbacks: {
				onFlowComplete: (t?: ChatMessageTimings) => {
					final = t;
				}
			},
			conversationId: 'conv-w2',
			messages: [{ content: 'go', role: 'user' }],
			options: { model: 'm' },
			tools: [{ function: { description: '', name: 'shot', parameters: {} }, type: 'function' }]
		});

		const rec = final!.agentic!.toolCalls![0];

		expect(rec).toMatchObject({
			args_bytes: 10,
			error: 'Error: boom',
			result_bytes: 11,
			success: false,
			toolCallId: 'call_9'
		});
		expect(rec.t_end! - rec.t_start!).toBe(rec.duration_ms);
		// D3: draft fields of the last request kept; flow sums carry cache + draft
		expect(final).toMatchObject({ draft_n: 20, draft_n_accepted: 15 });
		expect(final!.agentic!.llm).toMatchObject({ cache_n: 1900, draft_n: 30, draft_n_accepted: 22 });
		expect(final!.agentic!.t_start).toBeLessThanOrEqual(final!.agentic!.t_end!);
	}, 30000);
});

describe('W2 client timings', () => {
	it('the chat service stamps request start, TTFT and wall on the timings', async () => {
		vi.stubGlobal(
			'fetch',
			vi.fn(async () =>
				sse([
					{ choices: [{ delta: { content: 'hi' } }] },
					{
						choices: [{ delta: {}, finish_reason: 'stop' }],
						timings: { predicted_ms: 5, predicted_n: 1, prompt_ms: 1, prompt_n: 1 }
					}
				])
			)
		);

		let got: ChatMessageTimings | undefined;

		await ChatService.sendMessage([{ content: 'x', role: 'user' }], {
			onComplete: (_r: string, _rc?: string, t?: ChatMessageTimings) => {
				got = t;
			},
			stream: true
		});

		expect(got?.client?.t_start).toBeGreaterThan(0);
		expect(got?.client?.ttft_ms).toBeGreaterThanOrEqual(0);
		expect(got!.client!.wall_ms).toBeGreaterThanOrEqual(got!.client!.ttft_ms!);
		expect(got?.predicted_n).toBe(1);
	});
});

describe('W2 D4 + context-bar line', () => {
	const flow = (prompt: number, cache: number, out: number, ms: number, tools: number) => ({
		agentic: {
			llm: { cache_n: cache, predicted_ms: ms, predicted_n: out, prompt_ms: ms, prompt_n: prompt },
			toolCallsCount: 1,
			toolsMs: tools,
			turns: 2
		}
	});
	const chat = [
		msg(MessageRole.USER, undefined, { timestamp: 1000 }),
		msg(MessageRole.ASSISTANT, flow(100, 1000, 10, 100, 500)),
		msg(MessageRole.ASSISTANT, flow(300, 3000, 30, 300, 900)),
		msg(MessageRole.USER),
		msg(MessageRole.ASSISTANT, {
			cache_n: 5000,
			draft_n: 10,
			draft_n_accepted: 8,
			predicted_ms: 400,
			predicted_n: 20,
			prompt_ms: 200,
			prompt_n: 50
		}),
		msg(MessageRole.USER),
		msg(MessageRole.ASSISTANT, flow(10, 100, 1, 10, 50))
	];

	it('D4: across all turns counts every flow once (its last snapshot) incl. cache, and plain replies', () => {
		expect(cumulativeUsage(chat)).toEqual({
			cacheTotal: 3000 + 5000 + 100,
			output: 30 + 20 + 1,
			outputMs: 300 + 400 + 10,
			promptMs: 300 + 200 + 10,
			read: 300 + 50 + 10,
			toolsMs: 900 + 50
		});
	});

	it('the bar line: last request speeds + MTP, compute = model + tools, wall only as detail', () => {
		const line = sessionLine(chat.slice(0, 5), 61000);

		expect(line.prefillTps).toBe(250);
		expect(line.decodeTps).toBe(50);
		expect(line.mtp).toBeCloseTo(0.8);
		expect(line.modelMs).toBe(300 + 300 + 200 + 400);
		expect(line.toolsMs).toBe(900);
		expect(line.sessionMs).toBe(60000);
		expect(formatClock(65_000)).toBe('1:05');
		expect(formatClock(3_725_000)).toBe('1:02:05');
	});
});
