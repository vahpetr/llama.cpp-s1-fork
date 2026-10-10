import { afterEach, beforeAll, describe, expect, it, vi } from 'vitest';

vi.mock('$app/navigation', () => ({ goto: vi.fn(async () => {}) }));

// node env unit project has no DOM, install a minimal localStorage backed by a Map
beforeAll(() => {
	const store = new Map<string, string>();
	const polyfill: Storage = {
		clear: () => store.clear(),
		getItem: (k) => (store.has(k) ? store.get(k)! : null),
		key: (i) => Array.from(store.keys())[i] ?? null,
		get length() {
			return store.size;
		},
		removeItem: (k) => {
			store.delete(k);
		},
		setItem: (k, v) => {
			store.set(k, String(v));
		}
	};

	(globalThis as unknown as { localStorage: Storage }).localStorage = polyfill;
});

import { CONTEXT_RESERVE } from '$lib/constants';
import { MessageRole } from '$lib/enums';
import { ChatService } from '$lib/services/chat.service';
import { DatabaseService } from '$lib/services/database.service';
import { clearImEndCacheForTest } from '$lib/services/im-end-token';
import { agenticStore } from '$lib/stores/agentic/index.svelte';
import { draftMessagesStore } from '$lib/stores/chat/drafts.svelte';
import { chatStore } from '$lib/stores/chat/index.svelte';
import { conversationsStore } from '$lib/stores/conversations/index.svelte';
import { modelsStore } from '$lib/stores/models/index.svelte';
import { settingsStore } from '$lib/stores/settings/index.svelte';
import type { DatabaseConversation, DatabaseMessage } from '$lib/types/database';
import { filterByLeafNodeId } from '$lib/utils';
import {
	applyRequestCeilings,
	cappedThinkingBudget,
	compactedConversationName,
	contextBudget,
	estimateMessageTokens,
	joinWithOverlap,
	legacyContextFields,
	measureContextUsage,
	normalizeContextOverrides,
	planSummaryRequest,
	replyTokenCap,
	resolveContextReserveConfig,
	selectTailMessages,
	shouldCompactBeforeTurn,
	summaryLengthUsable,
	summaryRejection,
	tailTokenBudget,
	unfinishedTail
} from '$lib/utils/context-reserve';

let seq = 0;

const SUMMARY = 'SUMMARY TEXT: ' + 'the state of the work and the next steps. '.repeat(4);

function msg(role: MessageRole, content: string, extra: Partial<DatabaseMessage> = {}) {
	return {
		children: [],
		content,
		convId: 'c1',
		id: `m${++seq}`,
		parent: null,
		role,
		timestamp: seq,
		toolCalls: '',
		type: role === MessageRole.SYSTEM ? 'system' : 'text',
		...extra
	} as DatabaseMessage;
}

const t = (prompt_n: number, cache_n: number, predicted_n: number) => ({
	cache_n,
	predicted_n,
	prompt_n
});

/** an SSE body; `finish` null = no finish chunk, `done` false = no [DONE] */
function sse(
	content: string,
	o: {
		finish?: string | null;
		done?: boolean;
		toolCall?: boolean;
		timings?: object;
	} = {}
): Response {
	const finish = o.finish === undefined ? 'stop' : o.finish;
	const call = { function: { arguments: '{}', name: 'x' }, id: 'c', index: 0, type: 'function' };
	const chunks = [
		`data: ${JSON.stringify({ choices: [{ delta: { content } }] })}\n\n`,
		o.toolCall
			? `data: ${JSON.stringify({ choices: [{ delta: { tool_calls: [call] } }] })}\n\n`
			: '',
		finish || o.timings
			? `data: ${JSON.stringify({ choices: [{ delta: {}, finish_reason: finish }], timings: o.timings })}\n\n`
			: '',
		o.done === false ? '' : 'data: [DONE]\n\n'
	];

	return new Response(chunks.join(''), { headers: { 'Content-Type': 'text/event-stream' } });
}

// ── a fake database + the conversation store over it ───────────────────────────────────────────

type Respond = (body: Record<string, unknown>, n: number, url: string) => Response;

function harness(
	nCtx: number,
	rows: DatabaseMessage[],
	conv: DatabaseConversation,
	respond: Respond
) {
	const convs = new Map<string, DatabaseConversation>([[conv.id, structuredClone(conv)]]);
	const root = {
		...msg(MessageRole.SYSTEM, ''),
		convId: conv.id,
		id: 'root',
		type: 'root'
	} as DatabaseMessage;
	const db = new Map<string, DatabaseMessage>(
		[root, ...rows].map((m) => [m.id, structuredClone(m)])
	);
	const bodies: Record<string, unknown>[] = [];
	const urls: string[] = [];

	let nextId = 0;

	const pathOf = (convId: string) =>
		(
			filterByLeafNodeId(
				[...db.values()].filter((m) => m.convId === convId),
				convs.get(convId)!.currNode!,
				false
			) as DatabaseMessage[]
		).filter((m) => m.type !== 'root');

	vi.spyOn(DatabaseService, 'getConversation').mockImplementation(async (id) =>
		structuredClone(convs.get(id))
	);
	vi.spyOn(DatabaseService, 'getConversationMessages').mockImplementation(async (id) =>
		[...db.values()].filter((m) => m.convId === id).map((m) => structuredClone(m))
	);
	vi.spyOn(DatabaseService, 'createConversation').mockImplementation(async (name, fields) => {
		const c = { currNode: '', id: `conv-${++nextId}`, lastModified: 1, name, ...fields };

		convs.set(c.id, c as DatabaseConversation);

		return c as DatabaseConversation;
	});
	vi.spyOn(DatabaseService, 'createRootMessage').mockImplementation(async (convId) => {
		const id = `root-${++nextId}`;

		db.set(id, { ...msg(MessageRole.SYSTEM, ''), convId, id, type: 'root' } as DatabaseMessage);

		return id;
	});
	vi.spyOn(DatabaseService, 'createSystemMessage').mockImplementation(
		async (convId, content, parent) => {
			const m = { ...msg(MessageRole.SYSTEM, content), convId, id: `s-${++nextId}`, parent };

			db.set(m.id, m);

			return m;
		}
	);
	vi.spyOn(DatabaseService, 'createMessageBranch').mockImplementation(async (m, parent) => {
		const created = { ...(m as DatabaseMessage), id: `n-${++nextId}`, parent };

		db.set(created.id, structuredClone(created));
		convs.get(created.convId)!.currNode = created.id;

		return created;
	});
	vi.spyOn(DatabaseService, 'updateConversation').mockImplementation(async (id, updates) => {
		convs.set(id, { ...convs.get(id)!, ...updates });
	});
	vi.spyOn(DatabaseService, 'updateMessage').mockImplementation(async (id, updates) => {
		const row = db.get(id);

		if (row) db.set(id, structuredClone({ ...row, ...updates }));
	});
	vi.spyOn(DatabaseService, 'updateCurrentNode').mockResolvedValue();
	vi.spyOn(conversationsStore, 'loadConversations').mockResolvedValue();
	vi.spyOn(conversationsStore, 'loadConversation').mockImplementation(async (id) => {
		conversationsStore.activeConversation = structuredClone(convs.get(id)!);
		conversationsStore.activeMessages = pathOf(id).map((m) => structuredClone(m));

		return true;
	});
	vi.spyOn(modelsStore.props, 'getModelContextSize').mockReturnValue(nCtx);
	vi.spyOn(modelsStore, 'activeModelId', 'get').mockReturnValue('m');
	vi.spyOn(agenticStore, 'getConfig').mockReturnValue({ enabled: false } as never);
	vi.spyOn(agenticStore, 'runAgenticFlow').mockResolvedValue({ handled: false });

	vi.stubGlobal(
		'fetch',
		vi.fn(async (url: string, init?: RequestInit) => {
			const body = JSON.parse(String(init?.body ?? '{}'));

			urls.push(String(url));
			bodies.push(body);

			return respond(body, bodies.length, String(url));
		})
	);

	conversationsStore.activeConversation = structuredClone(conv);
	conversationsStore.activeMessages = rows.map((m) => structuredClone(m));

	return { bodies, convs, db, pathOf, urls };
}

function cleanup() {
	vi.restoreAllMocks();
	vi.unstubAllGlobals();
	conversationsStore.activeConversation = null;
	conversationsStore.activeMessages = [];
	conversationsStore.conversations = [];
	chatStore.setChatLoading('c1', false);
	chatStore.clearChatStreaming('c1');
	clearImEndCacheForTest();
}

function setConfig(o: Record<string, unknown> = {}) {
	settingsStore.config = {
		...settingsStore.config,
		autoCompact: true,
		compactSummaryPrompt: '',
		compactSummaryTokens: 1000,
		compactTailTokens: 30,
		customJson: '{"tag":"w1"}',
		max_tokens: -1,
		...o
	};
}

// ── pure rules ─────────────────────────────────────────────────────────────────────────────────

describe('W1.2 derived budget (the pi rules)', () => {
	it('reserve = summary + instruction + slack (10 % of used, min 2048); pre-turn = reserve + 32768', () => {
		const p = { summaryPrompt: 'Summarize in {{MAX_TOKENS}}.', summaryTokens: 8192 };
		const big = contextBudget({ ...p, nCtx: 1048576, used: 500000 });
		const instr = Math.ceil('Summarize in 8192.'.length / 3) + 16;

		expect(big).toEqual({
			instructionTokens: instr,
			preTurnMin: 8192 + instr + 50000 + 32768,
			reserve: 8192 + instr + 50000,
			slackTokens: 50000,
			summaryTokens: 8192
		});
		// the 2048 slack floor
		expect(contextBudget({ ...p, nCtx: 524288, used: 1000 }).slackTokens).toBe(2048);
		// small contexts: summary <= n_ctx / 8, slack floor n_ctx / 16, reply room n_ctx / 4
		const small = contextBudget({ ...p, nCtx: 16384, used: 1000 });

		expect(small.summaryTokens).toBe(2048);
		expect(small.slackTokens).toBe(1024);
		expect(small.preTurnMin - small.reserve).toBe(4096);
	});

	it('rule 1: a user turn compacts first below the pre-turn minimum', () => {
		expect(shouldCompactBeforeTurn({ nCtx: 524288, preTurnMin: 81920, used: 442000 })).toBe(false);
		expect(shouldCompactBeforeTurn({ nCtx: 524288, preTurnMin: 81920, used: 442369 })).toBe(true);
		expect(
			shouldCompactBeforeTurn({ nCtx: 524288, pending: 500, preTurnMin: 81920, used: 442000 })
		).toBe(true);
		expect(shouldCompactBeforeTurn({ nCtx: null, preTurnMin: 81920, used: 999999 })).toBe(false);
	});

	it('reply cap = min(configured, n_ctx - used - reserve), floor 1', () => {
		expect(replyTokenCap({ nCtx: 10000, reserve: 2000, used: 5000 })).toBe(3000);
		expect(replyTokenCap({ configured: 1000, nCtx: 10000, reserve: 2000, used: 5000 })).toBe(1000);
		expect(replyTokenCap({ nCtx: null, reserve: 2000, used: 5000 })).toBeUndefined();
		expect(replyTokenCap({ nCtx: 10000, reserve: 2000, used: 9000 })).toBe(1);
	});

	it('a summary length below the 512 minimum is unusable', () => {
		expect(summaryLengthUsable(512)).toBe(true);
		expect(summaryLengthUsable(511)).toBe(false);
	});

	it("measures with the server's count and estimates what followed it (+10 %)", () => {
		const a = msg(MessageRole.ASSISTANT, 'x', { timings: t(100, 900, 50) });
		const u = msg(MessageRole.USER, 'y'.repeat(300));

		expect(measureContextUsage([a], 2000)).toEqual({ full: false, measured: 1050, used: 1050 });
		expect(measureContextUsage([a, u], 2000).used).toBe(1050 + Math.ceil((100 + 8) * 1.1));
		expect(measureContextUsage([a], 1051).full).toBe(true);
	});

	it('names a continuation', () => {
		expect(compactedConversationName('Plan')).toBe('Plan (cont.)');
		expect(compactedConversationName('Plan (cont.)')).toBe('Plan (cont. 2)');
		expect(compactedConversationName('Plan (cont. 2)')).toBe('Plan (cont. 3)');
	});
});

describe('summary request plan', () => {
	const sys = msg(MessageRole.SYSTEM, 'be brief');
	const u1 = msg(MessageRole.USER, 'q1');
	const a1 = msg(MessageRole.ASSISTANT, 'r1', { timings: t(10, 0, 990) });
	const u2 = msg(MessageRole.USER, 'q2');
	const plan = (messages: DatabaseMessage[], extra: Record<string, unknown> = {}) =>
		planSummaryRequest({
			instructionTokens: 100,
			messages,
			nCtx: 10000,
			summaryTokens: 1900,
			...extra
		});

	it('sends the whole conversation when the summary fits', () => {
		const a2 = msg(MessageRole.ASSISTANT, 'r2', { timings: t(20, 1000, 2000) });

		expect(plan([sys, u1, a1, u2, a2])).toEqual({
			cut: false,
			keep: 5,
			maxTokens: 1900,
			promptTokens: 3020
		});
	});

	it('the 100 %-full path: cut before the truncated reply at its measured prompt size', () => {
		const a2 = msg(MessageRole.ASSISTANT, 'r2 ...', { timings: t(20, 1000, 8979) });

		expect(plan([sys, u1, a1, u2, a2])).toEqual({
			cut: true,
			keep: 4,
			maxTokens: 1900,
			promptTokens: 1020
		});
	});

	it('rule 2: a partial reply is always left out of the summary request', () => {
		const a2 = msg(MessageRole.ASSISTANT, 'partial', { timings: t(20, 1000, 500) });

		expect(plan([sys, u1, a1, u2, a2], { cutBeforeLast: true })).toMatchObject({
			cut: true,
			keep: 4
		});
	});

	it('skips a non-prefix measurement (the first message of an agentic flow)', () => {
		const flowFirst = msg(MessageRole.ASSISTANT, '', { timings: t(9000, 500, 400) });
		const tool = msg(MessageRole.TOOL, 'out');
		const flowLast = msg(MessageRole.ASSISTANT, 'done', { timings: t(9000, 500, 499) });

		expect(plan([sys, u1, a1, u2, flowFirst, tool, flowLast])).toMatchObject({
			cut: true,
			keep: 2,
			promptTokens: 10
		});
	});

	it('null when no point leaves room, or the summary length is below the minimum', () => {
		const a2 = msg(MessageRole.ASSISTANT, 'r2', { timings: t(9900, 0, 99) });

		expect(plan([u2, a2])).toBeNull();
		expect(plan([u1, a1], { nCtx: 100000, summaryTokens: 400 })).toBeNull();
	});
});

describe('W1.2 token-based tail', () => {
	const u1 = msg(MessageRole.USER, 'a'.repeat(300));
	const call = msg(MessageRole.ASSISTANT, '', { toolCalls: '[{"id":"1"}]' });
	const tool1 = msg(MessageRole.TOOL, 'r'.repeat(250));
	const fin = msg(MessageRole.ASSISTANT, 'done');
	const body = [u1, call, tool1, fin];
	const cost = body.map(estimateMessageTokens);

	it('keeps the newest whole messages that fit the token budget', () => {
		expect(
			selectTailMessages(
				body,
				cost.reduce((a, b) => a + b, 0)
			)
		).toEqual(body);
		expect(selectTailMessages(body, cost[1] + cost[2] + cost[3])).toEqual([call, tool1, fin]);
		expect(selectTailMessages(body, 0)).toEqual([]);
	});

	it('never starts with a tool result', () => {
		expect(selectTailMessages(body, cost[2] + cost[3])).toEqual([fin]);
	});

	it('the budget is at most a quarter of the context and what the new chat has left', () => {
		expect(
			tailTokenBudget({ headTokens: 100, nCtx: 8192, reserve: 6144, summaryTokens: 1000 })
		).toBe(8192 - 6144 - 1000 - 100 - 256);
		expect(
			tailTokenBudget({ headTokens: 0, nCtx: 100000, reserve: 6144, summaryTokens: 2000 })
		).toBe(25000);
		expect(tailTokenBudget({ headTokens: 0, nCtx: 4000, reserve: 3900, summaryTokens: 500 })).toBe(
			0
		);
	});
});

describe('rule 3: the unfinished-reply detector (pi cases)', () => {
	it('flags unfinished tails', () => {
		expect(unfinishedTail('Here is the fix:\n\n```ts\nconst a = 1;\n')).toBe('unclosed code fence');
		expect(unfinishedTail('x '.repeat(2000) + '\n```py\nprint(1)')).toBe('unclosed code fence');
		expect(unfinishedTail("Now I'll update the file:")).not.toBeNull();
		expect(unfinishedTail('Сейчас обновлю файл:')).not.toBeNull();
		expect(unfinishedTail('The result is —')).not.toBeNull();
		expect(unfinishedTail('| a | b |\n|---|---|\n| 1 | 2')).not.toBeNull();
		expect(unfinishedTail('Steps:\n1. build\n2.')).not.toBeNull();
		expect(unfinishedTail('Plan\n- ')).not.toBeNull();
	});

	it('leaves legitimate endings alone (a long reply ending in ":" is prose)', () => {
		const prose = 'This explains the trade-off in detail and lists the options considered. '.repeat(
			4
		);

		expect(unfinishedTail(prose + 'Two things remain for the user to decide:')).toBeNull();
		expect(unfinishedTail('Done.')).toBeNull();
		expect(unfinishedTail('Done — all green.')).toBeNull();
		expect(unfinishedTail('```ts\nconst a = 1;\n```\nThat is all.')).toBeNull();
		expect(unfinishedTail('Ratio is 3:4.')).toBeNull();
		expect(unfinishedTail('See 10:30')).toBeNull();
		expect(unfinishedTail('| a | b |\n|---|---|\n| 1 | 2 |')).toBeNull();
		expect(unfinishedTail('')).toBeNull();
	});

	it('joins with an overlap check', () => {
		const A = 'function sub(a, b) {\n  return a';

		expect(joinWithOverlap(A, ' - b;\n}\n')).toEqual({ overlap: 0, text: A + ' - b;\n}\n' });

		const rep = joinWithOverlap(A, 'function sub(a, b) {\n  return a - b;\n}\n');

		expect(rep.overlap).toBeGreaterThanOrEqual(24);
		expect(rep.text).toBe(A + ' - b;\n}\n');
		expect(joinWithOverlap('ab', 'abc').overlap).toBe(0);
	});
});

describe('W1.2 settings: per-chat overrides and migration', () => {
	afterEach(cleanup);

	const global = {
		autoCompact: true,
		compactSummaryPrompt: 'GLOBAL',
		compactSummaryTokens: 8192,
		compactTailTokens: 16384
	};

	it('defaults: summary 8192, tail 16384', () => {
		expect(resolveContextReserveConfig({}, undefined)).toMatchObject({
			autoCompact: true,
			summaryTokens: 8192,
			tailTokens: 16384
		});
	});

	it('each field of the chat overrides its global setting; invalid values fall back', () => {
		expect(
			resolveContextReserveConfig(global, {
				autoCompact: false,
				summaryPrompt: 'MINE',
				tailTokens: 0
			})
		).toEqual({ autoCompact: false, summaryPrompt: 'MINE', summaryTokens: 8192, tailTokens: 0 });
		expect(
			resolveContextReserveConfig(global, { summaryPrompt: ' ', summaryTokens: 0 })
		).toMatchObject({ summaryPrompt: 'GLOBAL', summaryTokens: 8192 });
	});

	it('migration: an old reserve / keepMessages is ignored, reported, dropped on save', () => {
		const old = { keepMessages: 2, reserve: 3000, summaryPrompt: 'MINE' };

		expect(resolveContextReserveConfig(global, old)).toEqual({
			autoCompact: true,
			summaryPrompt: 'MINE',
			summaryTokens: 8192,
			tailTokens: 16384
		});
		expect(legacyContextFields(global, old)).toEqual(['reserve', 'messages kept']);
		expect(legacyContextFields({ ...global, contextReserveTokens: 6144 }, undefined)).toEqual([
			'reserve'
		]);
		expect(legacyContextFields(global, undefined)).toEqual([]);
		expect(normalizeContextOverrides(old)).toEqual({ summaryPrompt: 'MINE' });
		expect(normalizeContextOverrides({ reserve: 1 })).toBeUndefined();
	});

	it('the store resolves the active chat and listed chats', () => {
		settingsStore.config = { ...settingsStore.config, ...global };
		conversationsStore.activeConversation = {
			contextSettings: { summaryTokens: 111 },
			currNode: null,
			id: 'act',
			lastModified: 0,
			name: 'a'
		};
		conversationsStore.conversations = [
			{
				contextSettings: { tailTokens: 9 },
				currNode: null,
				id: 'other',
				lastModified: 0,
				name: 'o'
			}
		];

		expect(conversationsStore.getContextConfig('act').summaryTokens).toBe(111);
		expect(conversationsStore.getContextConfig('other')).toMatchObject({
			summaryTokens: 8192,
			tailTokens: 9
		});
		expect(conversationsStore.getContextConfig('nope').tailTokens).toBe(16384);
	});
});

// ── compaction against the cached prompt ───────────────────────────────────────────────────────

describe('W1: compaction against the cached prompt', () => {
	afterEach(cleanup);

	const N_CTX = 20000;
	const conv = { currNode: 'a2', id: 'c1', lastModified: 0, name: 'Plan' } as DatabaseConversation;

	function rowsOf(full: boolean): DatabaseMessage[] {
		return [
			{ ...msg(MessageRole.SYSTEM, 'be brief'), id: 'sys', parent: 'root' },
			{ ...msg(MessageRole.USER, 'question one'), id: 'u1', parent: 'sys' },
			{
				...msg(MessageRole.ASSISTANT, 'answer one', { timings: t(30, 0, 970) }),
				id: 'a1',
				parent: 'u1'
			},
			{ ...msg(MessageRole.USER, 'question two'), id: 'u2', parent: 'a1' },
			{
				...msg(MessageRole.ASSISTANT, 'answer two', {
					timings: full ? t(20, 1000, N_CTX - 1021) : t(20, 1000, 9000)
				}),
				id: 'a2',
				parent: 'u2'
			}
		];
	}

	async function scenario(
		full: boolean,
		o: {
			contextSettings?: DatabaseConversation['contextSettings'];
			summary?: () => Response;
			config?: Record<string, unknown>;
		} = {}
	) {
		setConfig(o.config);

		const rows = rowsOf(full);
		const h = harness(N_CTX, rows, { ...conv, contextSettings: o.contextSettings }, (_b, n) =>
			n === 1 && o.summary ? o.summary() : sse(n === 1 ? SUMMARY : 'reply')
		);
		const result = await chatStore.compactConversation('c1');

		return { ...h, result, rows };
	}

	it('the summary request = the next turn with an instruction instead of the user message', async () => {
		const { bodies, convs, pathOf, result, rows } = await scenario(false);

		expect(result).toEqual({ convId: 'conv-1', partialId: undefined });

		// the next real turn of the old conversation, as the chat store sends it
		conversationsStore.activeConversation = structuredClone({ ...conv });
		conversationsStore.activeMessages = rows;

		const userNew = { ...msg(MessageRole.USER, 'question three'), id: 'u3', parent: 'a2' };
		const assistantNew = { ...msg(MessageRole.ASSISTANT, ''), id: 'a3', parent: 'u3' };

		await chatStore.streamChatCompletion([...rows, userNew], assistantNew);

		const [summary, next] = bodies;
		const sumMessages = summary.messages as Record<string, unknown>[];
		const rest = (b: Record<string, unknown>) => {
			const r = { ...b };

			for (const k of [
				'messages',
				'max_tokens',
				'thinking_budget_tokens',
				'reasoning_budget_tokens'
			]) {
				delete r[k];
			}

			return r;
		};

		expect(sumMessages.slice(0, -1)).toEqual((next.messages as unknown[]).slice(0, -1));
		expect(String(sumMessages.at(-1)!.content)).toContain('continuation summary');
		expect(rest(summary)).toEqual(rest(next));
		expect(summary).toMatchObject({ max_tokens: 1000, stream: true, tag: 'w1' });
		expect(summary.thinking_budget_tokens).toBe(250);

		// the next reply is capped at n_ctx - used - derived reserve, thinking <= half
		const used = 10020 + Math.ceil((Math.ceil('question three'.length / 3) + 8) * 1.1);
		const { reserve } = contextBudget({
			nCtx: N_CTX,
			summaryPrompt: CONTEXT_RESERVE.DEFAULT_SUMMARY_PROMPT,
			summaryTokens: 1000,
			used
		});

		expect(next.max_tokens).toBe(N_CTX - used - reserve);
		expect(next.thinking_budget_tokens).toBe(Math.floor((next.max_tokens as number) / 2));

		// system, summary note, the tail (30 tokens = the last 2 messages); linked, the old one kept
		expect(convs.get('conv-1')).toMatchObject({
			compactedFromConversationId: 'c1',
			name: 'Plan (cont.)'
		});
		expect(convs.get('c1')).toMatchObject({ compactedIntoConversationId: 'conv-1' });

		const path = pathOf('conv-1');

		expect(path.map((m) => [m.role, m.content.slice(0, 20)])).toEqual([
			['system', 'be brief'],
			['assistant', 'Summary of the earli'],
			['user', 'question two'],
			['assistant', 'answer two']
		]);
		expect(path[1].content).toContain('SUMMARY TEXT');
		expect(path[3].timings).toBeUndefined();
	}, 60000);

	it('the 100 %-full path drops the truncated reply from the summary request', async () => {
		const { bodies } = await scenario(true);
		const sumMessages = bodies[0].messages as Record<string, unknown>[];

		expect(sumMessages.map((m) => m.role)).toEqual(['system', 'user', 'assistant', 'user', 'user']);
		expect(sumMessages[3]).toMatchObject({ content: 'question two' });
	}, 60000);

	it("a chat's own instruction, summary length and tail; the continuation inherits them", async () => {
		const own = {
			summaryPrompt: 'CHAT RULE: summarize in {{MAX_TOKENS}}',
			summaryTokens: 2400,
			tailTokens: 15
		};
		const { bodies, convs, pathOf } = await scenario(false, { contextSettings: own });
		const last = (bodies[0].messages as Record<string, unknown>[]).at(-1)!;

		expect(String(last.content)).toBe('CHAT RULE: summarize in 2400');
		expect(bodies[0].max_tokens).toBe(2400);
		expect(convs.get('conv-1')!.contextSettings).toEqual(own);
		expect(pathOf('conv-1').map((m) => m.content.slice(0, 10))).toEqual([
			'be brief',
			'Summary of',
			'answer two'
		]);
	}, 60000);

	it('sol #3: custom JSON cannot lift the summary ceilings (all aliases)', async () => {
		const { bodies } = await scenario(false, {
			config: {
				customJson:
					'{"max_tokens":-1,"n_predict":99999,"max_completion_tokens":50000,"reasoning_budget_tokens":8192}'
			}
		});

		expect(bodies[0]).toMatchObject({
			max_tokens: 1000,
			reasoning_budget_tokens: 250,
			thinking_budget_tokens: 250
		});
		expect(bodies[0].n_predict).toBeUndefined();
		expect(bodies[0].max_completion_tokens).toBeUndefined();
	}, 60000);

	it('sol #4: a tool-call answer, a length-truncated or unfinished stream is no summary', async () => {
		const cases: [() => Response, RegExp][] = [
			[() => sse(SUMMARY, { finish: 'tool_calls', toolCall: true }), /tool call/],
			[() => sse(SUMMARY, { finish: 'length' }), /length limit/],
			[() => sse(SUMMARY, { done: false, finish: null }), /before completion/],
			[() => sse('ok'), /too short/]
		];

		for (const [summary, why] of cases) {
			const { convs, result } = await scenario(false, { summary });

			expect(result).toMatchObject({ error: expect.stringMatching(why) });
			expect(convs.size).toBe(1);
			expect(convs.get('c1')!.compactedIntoConversationId).toBeUndefined();
			cleanup();
		}
	}, 60000);

	it('a summary length below the minimum is refused before any request', async () => {
		const { bodies, result } = await scenario(false, { config: { compactSummaryTokens: 400 } });

		expect(result).toMatchObject({ error: expect.stringMatching(/raise the summary length/) });
		expect(bodies).toHaveLength(0);
	}, 60000);
});

// ── rules 2 / 3 end to end ─────────────────────────────────────────────────────────────────────

describe('W1.2 rule 2: a reply cut by the reserve cap is compacted and continued', () => {
	afterEach(cleanup);

	const N_CTX = 20000;
	const PARTIAL = 'Step 1 is done. Now the long part of the answer continues here and';
	const conv = { currNode: 'a2', id: 'c1', lastModified: 0, name: 'L' } as DatabaseConversation;

	function rows(): DatabaseMessage[] {
		return [
			{ ...msg(MessageRole.SYSTEM, 'be brief'), id: 'sys', parent: 'root' },
			{ ...msg(MessageRole.USER, 'question one'), id: 'u1', parent: 'sys' },
			{
				...msg(MessageRole.ASSISTANT, 'answer one', { timings: t(30, 0, 970) }),
				id: 'a1',
				parent: 'u1'
			},
			{ ...msg(MessageRole.USER, 'write it all'), id: 'u2', parent: 'a1' },
			{
				...msg(MessageRole.ASSISTANT, PARTIAL, { timings: t(20, 1000, 15000) }),
				id: 'a2',
				parent: 'u2'
			}
		];
	}

	it('new chat = system + summary + tail + the partial; continued from the stop point, joined', async () => {
		setConfig();

		const h = harness(N_CTX, rows(), conv, (_b, n) =>
			n === 1
				? sse(SUMMARY)
				: // the continuation repeats the partial's last 30 chars, then goes on
					sse(PARTIAL.slice(-30) + ' finishes the answer.', { timings: t(5, 600, 9) })
		);

		await chatStore.afterReply('c1', 'a2', { byReserve: true, finish: 'length', toolCalls: false });

		const [summary, cont] = h.bodies;
		const sumMsgs = summary.messages as Record<string, unknown>[];

		// the partial is not summarised
		expect(sumMsgs.map((m) => m.role)).toEqual(['system', 'user', 'assistant', 'user', 'user']);
		expect(JSON.stringify(sumMsgs)).not.toContain('Step 1 is done');

		const path = h.pathOf('conv-1');

		expect(path[1].content).toContain(CONTEXT_RESERVE.PARTIAL_NOTE);
		expect(path.at(-2)).toMatchObject({ content: 'write it all', role: 'user' });
		expect(path.at(-1)).toMatchObject({
			content: PARTIAL + ' finishes the answer.',
			role: 'assistant'
		});
		expect(path.at(-1)!.replyNote).toBeUndefined();
		// the continuation request: the new chat with the partial as the final assistant message
		expect(cont).toMatchObject({ add_generation_prompt: false, continue_final_message: true });
		expect((cont.messages as Record<string, unknown>[]).at(-1)).toMatchObject({
			content: PARTIAL,
			role: 'assistant'
		});
	}, 60000);

	it('still cut after 2 continuations: the message is marked truncated', async () => {
		setConfig();

		const h = harness(N_CTX, rows(), conv, (b) =>
			(b.messages as Record<string, unknown>[]).at(-1)?.role === 'user'
				? sse(SUMMARY)
				: sse(' more', { finish: 'length', timings: t(5, 600, 9) })
		);

		await chatStore.afterReply('c1', 'a2', { byReserve: true, finish: 'length', toolCalls: false });

		// 2 x (summary + continuation)
		expect(h.bodies).toHaveLength(4);

		const second = [...h.convs.values()].find((c) => c.compactedFromConversationId === 'conv-1')!;
		const last = h.pathOf(second.id).at(-1)!;

		expect(last.replyNote).toMatch(/reply truncated: still cut/);
		expect(last.content).toBe(PARTIAL + ' more more');
	}, 60000);

	it('a `length` stop at the configured max_tokens (not our cap) is left alone', async () => {
		setConfig();

		const h = harness(N_CTX, rows(), conv, () => sse(SUMMARY));

		await chatStore.afterReply('c1', 'a2', {
			byReserve: false,
			finish: 'length',
			toolCalls: false
		});
		expect(h.bodies).toHaveLength(0);
	});
});

describe('W1.2 rule 3: the unfinished-reply guard', () => {
	afterEach(cleanup);

	function setup(reply: string, respond: Respond) {
		setConfig();

		return harness(
			100000,
			[
				{ ...msg(MessageRole.USER, 'fix it'), id: 'u1', parent: 'root' },
				{
					...msg(MessageRole.ASSISTANT, reply, { timings: t(10, 100, 12) }),
					id: 'a1',
					parent: 'u1'
				}
			],
			{ currNode: 'a1', id: 'c1', lastModified: 0, name: 'G' },
			respond
		);
	}

	const tokenize = () => new Response(JSON.stringify({ tokens: [248046] }));

	it('one continuation with -2.0 on <|im_end|>, joined', async () => {
		const h = setup("Now I'll update the file:", (_b, _n, url) =>
			url.includes('tokenize') ? tokenize() : sse('\n\n```ts\nconst x = 1;\n```\nDone.')
		);

		await chatStore.afterReply('c1', 'a1', { byReserve: true, finish: 'stop', toolCalls: false });

		expect(h.urls[0]).toContain('tokenize');
		expect(h.bodies[1].logit_bias).toEqual({ '248046': -2 });
		expect(h.bodies[1]).toMatchObject({ continue_final_message: true });
		expect(h.db.get('a1')!.content).toBe(
			"Now I'll update the file:\n\n```ts\nconst x = 1;\n```\nDone."
		);
		expect(h.db.get('a1')!.replyNote).toBeUndefined();
	}, 60000);

	it('<= 2 continuations; still unfinished = marked', async () => {
		const h = setup('First:', (_b, _n, url) =>
			url.includes('tokenize') ? tokenize() : sse(' and next:')
		);

		await chatStore.afterReply('c1', 'a1', { byReserve: true, finish: 'stop', toolCalls: false });

		expect(h.bodies.filter((b) => b.continue_final_message)).toHaveLength(2);
		expect(h.db.get('a1')!.replyNote).toMatch(/looks unfinished: short reply ending in ':'/);
	}, 60000);

	it('finished replies, tool-call replies and length stops are untouched', async () => {
		const h = setup('All done.', () => sse('x'));

		await chatStore.afterReply('c1', 'a1', { byReserve: true, finish: 'stop', toolCalls: false });
		await chatStore.afterReply('c1', 'a1', {
			byReserve: false,
			finish: 'length',
			toolCalls: false
		});
		await chatStore.afterReply('c1', 'a1', { byReserve: true, finish: 'stop', toolCalls: true });
		expect(h.bodies).toHaveLength(0);
	});
});

// ── sol review hardening ───────────────────────────────────────────────────────────────────────

describe('W1 hardening (sol review)', () => {
	afterEach(cleanup);

	function setupFullChat(respond: Respond) {
		setConfig({ customJson: '' });

		return harness(
			10000,
			[
				{ ...msg(MessageRole.USER, 'q1'), id: 'u1', parent: 'root' },
				{
					...msg(MessageRole.ASSISTANT, 'r1', { timings: t(10, 5000, 3000) }),
					id: 'a1',
					parent: 'u1'
				}
			],
			{ currNode: 'a1', id: 'c1', lastModified: 0, name: 'A' },
			respond
		);
	}

	it('#1: a failed auto-compaction after switching to B sends nothing into B; A keeps the draft', async () => {
		const h = setupFullChat(() => {
			conversationsStore.activeConversation = {
				currNode: null,
				id: 'b',
				lastModified: 0,
				name: 'B'
			};
			conversationsStore.activeMessages = [];

			return sse('Let me check.', { finish: 'tool_calls', toolCall: true });
		});
		const created = vi.mocked(DatabaseService.createMessageBranch);

		expect(chatStore.sendGate('c1', 'my question')).toBe('compact');
		await chatStore.sendMessage('my question');

		expect(h.bodies).toHaveLength(1);
		expect(created).not.toHaveBeenCalled();
		expect(draftMessagesStore.getDraftMessage('c1').message).toBe('my question');
		expect(draftMessagesStore.getDraftMessage('b').message).toBe('');
		draftMessagesStore.clearDraftMessage('c1');
	});

	it('#2: ownership is taken synchronously: a second compaction cannot slip in', async () => {
		let release: () => void = () => {};

		const gate = new Promise<void>((r) => (release = r));

		setupFullChat(() => sse(''));
		vi.stubGlobal(
			'fetch',
			vi.fn(async () => {
				await gate;

				return new Response('data: [DONE]\n\n', {
					headers: { 'Content-Type': 'text/event-stream' }
				});
			})
		);

		const first = chatStore.compactConversation('c1');

		expect(chatStore.compaction).toMatchObject({ convId: 'c1' });
		expect(chatStore.isChatLoadingInternal('c1')).toBe(true);
		expect(await chatStore.compactConversation('c1')).toEqual({
			error: 'A compaction is already running'
		});
		release();
		expect(await first).toMatchObject({ error: expect.any(String) });
		expect(chatStore.compaction).toBeNull();
		expect(chatStore.isChatLoadingInternal('c1')).toBe(false);
	});

	it('#5: auto-compaction off + no room above the reserve = the send is refused', () => {
		setupFullChat(() => sse(''));
		settingsStore.config = { ...settingsStore.config, autoCompact: false };
		expect(chatStore.sendGate('c1', 'more')).toBe('blocked');
		settingsStore.config = { ...settingsStore.config, autoCompact: true };
		expect(chatStore.sendGate('c1', 'more')).toBe('compact');
	});

	it('#3: request ceilings apply after the custom JSON over every alias', async () => {
		const body: Record<string, unknown> = {
			max_completion_tokens: 900,
			max_tokens: -1,
			n_predict: 5000,
			reasoning_budget_tokens: 8192
		};

		applyRequestCeilings(body, { maxTokens: 1000, thinkingBudget: 250 });
		expect(body).toEqual({
			max_tokens: 900,
			reasoning_budget_tokens: 250,
			thinking_budget_tokens: 250
		});

		const bodies: Record<string, unknown>[] = [];

		vi.stubGlobal(
			'fetch',
			vi.fn(async (_u: string, init?: RequestInit) => {
				bodies.push(JSON.parse(String(init?.body)));

				return new Response('{}', { status: 500 });
			})
		);
		await ChatService.sendMessage([{ content: 'x', role: 'user' }], {
			custom: '{"max_tokens":-1,"thinking_budget_tokens":8192,"logit_bias":{"7":1}}',
			logitBias: { '9': -2 },
			maxTokensCeiling: 300,
			stream: false,
			thinkingBudgetCap: 150
		} as never).catch(() => {});

		expect(bodies[0]).toMatchObject({
			logit_bias: { '7': 1, '9': -2 },
			max_tokens: 300,
			thinking_budget_tokens: 150
		});
	});

	it('#4: summary validation', () => {
		const ok = { completed: true, finishReason: 'stop', text: SUMMARY };

		expect(summaryRejection(ok)).toBeNull();
		expect(summaryRejection({ ...ok, completed: false })).toMatch(/before completion/);
		expect(summaryRejection({ ...ok, toolCalls: '[{"id":"1"}]' })).toMatch(/tool call/);
		expect(summaryRejection({ ...ok, finishReason: 'length' })).toMatch(/length limit/);
		expect(summaryRejection({ ...ok, finishReason: undefined })).toMatch(/did not finish/);
		expect(summaryRejection({ ...ok, text: 'short' })).toMatch(/too short/);
	});

	it('#7: thinking of a capped reply is at most half the cap; none for a tiny cap', () => {
		expect(cappedThinkingBudget(undefined)).toBeUndefined();
		expect(cappedThinkingBudget(256)).toBe(128);
		expect(cappedThinkingBudget(255)).toBe(0);
		expect(cappedThinkingBudget(10001)).toBe(5000);
	});

	it('#8: tool results are estimated at 2.5 chars/token, every estimate +10 %', () => {
		const text = 'z'.repeat(250);

		expect(estimateMessageTokens(msg(MessageRole.TOOL, text))).toBe(Math.ceil((100 + 8) * 1.1));
		expect(estimateMessageTokens(msg(MessageRole.USER, text))).toBe(Math.ceil((84 + 8) * 1.1));
	});
});

describe('W1: later turns of an agentic flow', () => {
	afterEach(cleanup);

	it('cap at free - derived reserve after the tool results; 0 = stop the flow', () => {
		vi.spyOn(modelsStore.props, 'getModelContextSize').mockReturnValue(20000);
		setConfig();

		const cap = (
			agenticStore as unknown as {
				laterTurnMaxTokens: (t: object, m: object[], model: string) => number | undefined;
			}
		).laterTurnMaxTokens.bind(agenticStore);
		const turn = [
			{ content: 'q', role: 'user' },
			{ content: '', role: 'assistant', tool_calls: [] },
			{ content: 'x'.repeat(300), role: 'tool', tool_call_id: '1' }
		];
		// tool result at 2.5 chars/token + 10 %: ceil((120 + 8) * 1.1) = 141
		const used = 5000 + 141;
		const { reserve } = contextBudget({
			nCtx: 20000,
			summaryPrompt: CONTEXT_RESERVE.DEFAULT_SUMMARY_PROMPT,
			summaryTokens: 1000,
			used
		});

		expect(cap(t(100, 4800, 100), turn, 'm')).toBe(20000 - used - reserve);
		expect(cap(t(100, 17000, 100), turn, 'm')).toBe(0);
	});
});

// ── astra verification of W1.2 ─────────────────────────────────────────────────────────────────

describe('W1.2 fixes (astra verification)', () => {
	afterEach(cleanup);

	const N_CTX = 20000;
	const conv = { currNode: 'a2', id: 'c1', lastModified: 0, name: 'Plan' } as DatabaseConversation;
	const PARTIAL = 'Step 1 is done. Now the long part of the answer continues here and';

	function rows(last = 'answer two', timings = t(20, 1000, 9000)): DatabaseMessage[] {
		return [
			{ ...msg(MessageRole.SYSTEM, 'be brief'), id: 'sys', parent: 'root' },
			{ ...msg(MessageRole.USER, 'question one'), id: 'u1', parent: 'sys' },
			{
				...msg(MessageRole.ASSISTANT, 'answer one', { timings: t(30, 0, 970) }),
				id: 'a1',
				parent: 'u1'
			},
			{ ...msg(MessageRole.USER, 'question two'), id: 'u2', parent: 'a1' },
			{ ...msg(MessageRole.ASSISTANT, last, { timings }), id: 'a2', parent: 'u2' }
		];
	}

	const B = { currNode: null, id: 'b', lastModified: 0, name: 'B' } as DatabaseConversation;

	it('#1: a compaction finishing after the user moved to B links the continuation, B stays open', async () => {
		setConfig();
		harness(N_CTX, rows(), conv, () => {
			conversationsStore.activeConversation = structuredClone(B);
			conversationsStore.activeMessages = [];

			return sse(SUMMARY);
		});

		const { goto } = await import('$app/navigation');

		vi.mocked(goto).mockClear();

		const result = await chatStore.compactConversation('c1');

		expect(result).toMatchObject({ convId: 'conv-1' });
		expect(conversationsStore.activeConversation?.id).toBe('b');
		expect(goto).not.toHaveBeenCalled();
		expect(conversationsStore.loadConversation).not.toHaveBeenCalled();
	});

	it("#2: A's queued message is never sent into B (plain reply drain)", async () => {
		setConfig({ customJson: '' });

		const h = harness(N_CTX, rows(), conv, () => {
			conversationsStore.activeConversation = structuredClone(B);
			conversationsStore.activeMessages = [];

			return sse('reply');
		});
		const rs = rows();
		const userNew = { ...msg(MessageRole.USER, 'q3'), id: 'u3', parent: 'a2' };
		const assistantNew = { ...msg(MessageRole.ASSISTANT, ''), id: 'a3', parent: 'u3' };

		chatStore.injectPendingMessage('c1', 'queued in A');
		await chatStore.streamChatCompletion([...rs, userNew], assistantNew);

		for (let i = 0; i < 50 && !draftMessagesStore.getDraftMessage('c1').message; i++) {
			await new Promise((r) => setTimeout(r, 5));
		}

		expect(h.bodies).toHaveLength(1);
		expect(draftMessagesStore.getDraftMessage('c1').message).toBe('queued in A');
		expect(draftMessagesStore.getDraftMessage('b').message).toBe('');
		draftMessagesStore.clearDraftMessage('c1');
	});

	it("#2: A's steering message after an agentic flow is never sent into B", async () => {
		setConfig({ customJson: '' });

		const h = harness(N_CTX, rows(), conv, () => sse('x'));

		vi.mocked(agenticStore.runAgenticFlow).mockImplementation(async () => {
			conversationsStore.activeConversation = structuredClone(B);
			conversationsStore.activeMessages = [];

			return { handled: true };
		});
		vi.spyOn(agenticStore, 'consumePendingSteeringMessage').mockReturnValue({
			content: 'steer A'
		} as never);

		const assistantNew = { ...msg(MessageRole.ASSISTANT, ''), id: 'a3', parent: 'a2' };

		await chatStore.streamChatCompletion(rows(), assistantNew);

		expect(h.bodies).toHaveLength(0);
		expect(draftMessagesStore.getDraftMessage('c1').message).toBe('steer A');
		draftMessagesStore.clearDraftMessage('c1');
	});

	it('#3: input queued during a mid-turn compaction is sent once, in the new chat, after the continuation', async () => {
		setConfig({ customJson: '' });

		const h = harness(N_CTX, rows(PARTIAL, t(20, 1000, 15000)), conv, (b, n) => {
			if (n === 1) return sse(SUMMARY);

			if (b.continue_final_message) return sse(' finishes.', { timings: t(5, 600, 9) });

			return sse('follow-up answer');
		});

		chatStore.injectPendingMessage('c1', 'follow-up');
		await chatStore.afterReply('c1', 'a2', { byReserve: true, finish: 'length', toolCalls: false });

		for (let i = 0; i < 100 && h.bodies.length < 3; i++) await new Promise((r) => setTimeout(r, 5));

		expect(h.bodies).toHaveLength(3);
		expect(h.bodies[1].continue_final_message).toBe(true);

		const third = h.bodies[2].messages as Record<string, unknown>[];

		expect(third.at(-1)).toMatchObject({ content: 'follow-up', role: 'user' });
		// the continued reply precedes it, in the new chat
		expect(third.at(-2)).toMatchObject({ content: PARTIAL + ' finishes.', role: 'assistant' });
		expect(chatStore.consumePendingMessage('c1')).toBeNull();
		expect(chatStore.consumePendingMessage('conv-1')).toBeNull();
	}, 60000);

	it('#4: excluded reasoning stays on the prefill being continued, not on earlier messages', async () => {
		const bodies: Record<string, unknown>[] = [];

		vi.stubGlobal(
			'fetch',
			vi.fn(async (_u: string, init?: RequestInit) => {
				bodies.push(JSON.parse(String(init?.body)));

				return new Response('{}', { status: 500 });
			})
		);
		await ChatService.sendMessage(
			[
				{ content: 'q', role: 'user' },
				{ content: 'a', reasoning_content: 'old thought', role: 'assistant' },
				{ content: 'q2', role: 'user' },
				{ content: '', reasoning_content: 'thinking so far', role: 'assistant' }
			],
			{ continueFinalMessage: true, excludeReasoningFromContext: true, stream: false } as never
		).catch(() => {});

		const m = bodies[0].messages as Record<string, unknown>[];

		expect(m[1].reasoning_content).toBeUndefined();
		expect(m[3]).toMatchObject({ reasoning_content: 'thinking so far', role: 'assistant' });
	});

	it('#5: custom JSON cannot replace the continuation messages or flags', async () => {
		const bodies: Record<string, unknown>[] = [];

		vi.stubGlobal(
			'fetch',
			vi.fn(async (_u: string, init?: RequestInit) => {
				bodies.push(JSON.parse(String(init?.body)));

				return new Response('{}', { status: 500 });
			})
		);
		await ChatService.sendMessage(
			[
				{ content: 'q', role: 'user' },
				{ content: 'partial', role: 'assistant' }
			],
			{
				continueFinalMessage: true,
				custom:
					'{"continue_final_message":false,"add_generation_prompt":true,"messages":[{"role":"user","content":"x"}]}',
				stream: false
			} as never
		).catch(() => {});

		expect(bodies[0]).toMatchObject({ add_generation_prompt: false, continue_final_message: true });
		expect((bodies[0].messages as Record<string, unknown>[]).at(-1)).toMatchObject({
			content: 'partial',
			role: 'assistant'
		});
	});

	it('#6: with no measurement yet, the tool definitions count; a measured prompt already has them', () => {
		setConfig();
		vi.spyOn(agenticStore, 'getConfig').mockReturnValue({ enabled: true } as never);

		const tools = [{ function: { description: 'd'.repeat(5000), name: 'big' }, type: 'function' }];

		conversationsStore.activeConversation = {
			...conv,
			toolsSnapshot: JSON.stringify(tools)
		} as DatabaseConversation;

		const fresh = [msg(MessageRole.USER, 'hi')];
		const measured = rows();
		const toolsTokens = Math.ceil(Math.ceil(JSON.stringify(tools).length / 2.5) * 1.1);

		expect(chatStore.usedTokens(fresh, 20000, 'c1')).toBe(
			estimateMessageTokens(fresh[0]) + toolsTokens
		);
		expect(chatStore.usedTokens(measured, 20000, 'c1')).toBe(10020);
	});
});
