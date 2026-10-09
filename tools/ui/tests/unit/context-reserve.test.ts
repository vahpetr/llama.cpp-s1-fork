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

import { MessageRole } from '$lib/enums';
import { DatabaseService } from '$lib/services/database.service';
import { agenticStore } from '$lib/stores/agentic/index.svelte';
import { chatStore } from '$lib/stores/chat/index.svelte';
import { conversationsStore } from '$lib/stores/conversations/index.svelte';
import { modelsStore } from '$lib/stores/models/index.svelte';
import { settingsStore } from '$lib/stores/settings/index.svelte';
import type { DatabaseConversation, DatabaseMessage } from '$lib/types/database';
import {
	compactedConversationName,
	measureContextUsage,
	planSummaryRequest,
	replyTokenCap,
	selectTailMessages,
	shouldCompact
} from '$lib/utils/context-reserve';

let seq = 0;

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

describe('context reserve arithmetic', () => {
	it('caps a reply at free - reserve and keeps a smaller configured limit', () => {
		expect(replyTokenCap({ nCtx: 10000, reserve: 2000, used: 5000 })).toBe(3000);
		expect(replyTokenCap({ configured: 1000, nCtx: 10000, reserve: 2000, used: 5000 })).toBe(1000);
		// reserve off, context unknown or the reserve already used up: the configured value as is
		expect(replyTokenCap({ configured: 500, nCtx: 10000, reserve: 0, used: 5000 })).toBe(500);
		expect(replyTokenCap({ nCtx: null, reserve: 2000, used: 5000 })).toBeUndefined();
		expect(replyTokenCap({ nCtx: 10000, reserve: 2000, used: 9000 })).toBeUndefined();
	});

	it('compacts when the turn would leave less than the reserve (+ a minimal reply) free', () => {
		expect(shouldCompact({ nCtx: 10000, reserve: 2000, used: 7000 })).toBe(false);
		expect(shouldCompact({ nCtx: 10000, pending: 900, reserve: 2000, used: 7000 })).toBe(true);
		expect(shouldCompact({ nCtx: 10000, reserve: 2000, used: 7900 })).toBe(true);
		expect(shouldCompact({ nCtx: 10000, reserve: 0, used: 9999 })).toBe(false);
		expect(shouldCompact({ nCtx: null, reserve: 2000, used: 9999 })).toBe(false);
	});

	it("measures with the server's count and estimates what followed it", () => {
		const a = msg(MessageRole.ASSISTANT, 'x', { timings: t(100, 900, 50) });
		const u = msg(MessageRole.USER, 'y'.repeat(300));

		expect(measureContextUsage([a], 2000)).toEqual({ full: false, measured: 1050, used: 1050 });
		expect(measureContextUsage([a, u], 2000).used).toBe(1050 + 100 + 8);
		// a reply that ran into the end of the context
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

	it('sends the whole conversation when the summary fits', () => {
		const a2 = msg(MessageRole.ASSISTANT, 'r2', { timings: t(20, 1000, 2000) });
		const plan = planSummaryRequest({
			instructionTokens: 100,
			messages: [sys, u1, a1, u2, a2],
			nCtx: 10000,
			reserve: 2000
		});

		expect(plan).toEqual({ cut: false, keep: 5, maxTokens: 1900, promptTokens: 3020 });
	});

	it('a full context (truncated reply) is cut before that reply at its measured prompt size', () => {
		const a2 = msg(MessageRole.ASSISTANT, 'r2 ...', { timings: t(20, 1000, 8979) });
		const plan = planSummaryRequest({
			instructionTokens: 100,
			messages: [sys, u1, a1, u2, a2],
			nCtx: 10000,
			reserve: 2000
		});

		expect(plan).toEqual({ cut: true, keep: 4, maxTokens: 1900, promptTokens: 1020 });
	});

	it('skips a non-prefix measurement (the first message of an agentic flow)', () => {
		const flowFirst = msg(MessageRole.ASSISTANT, '', { timings: t(9000, 500, 400) });
		const tool = msg(MessageRole.TOOL, 'out');
		const flowLast = msg(MessageRole.ASSISTANT, 'done', { timings: t(9000, 500, 499) });
		const plan = planSummaryRequest({
			instructionTokens: 100,
			messages: [sys, u1, a1, u2, flowFirst, tool, flowLast],
			nCtx: 10000,
			reserve: 2000
		});

		// before flowLast leaves too little room; flowFirst carries the flow's last-turn count (not
		// below flowLast's), so it is no cut point; the next one is before a1
		expect(plan?.cut).toBe(true);
		expect(plan?.keep).toBe(2);
		expect(plan?.promptTokens).toBe(10);
	});

	it('returns null when no point leaves room', () => {
		const a2 = msg(MessageRole.ASSISTANT, 'r2', { timings: t(9900, 0, 99) });

		expect(
			planSummaryRequest({ instructionTokens: 100, messages: [u2, a2], nCtx: 10000, reserve: 2000 })
		).toBeNull();
	});
});

describe('tail selection', () => {
	const u1 = msg(MessageRole.USER, 'a');
	const call = msg(MessageRole.ASSISTANT, '', { toolCalls: '[{"id":"1"}]' });
	const tool1 = msg(MessageRole.TOOL, 'r1');
	const tool2 = msg(MessageRole.TOOL, 'r2');
	const fin = msg(MessageRole.ASSISTANT, 'done');

	it('keeps the last K and never starts with a tool result', () => {
		expect(selectTailMessages([u1, call, tool1, tool2, fin], 4, 1e6)).toEqual([
			call,
			tool1,
			tool2,
			fin
		]);
		expect(selectTailMessages([u1, call, tool1, tool2, fin], 3, 1e6)).toEqual([
			call,
			tool1,
			tool2,
			fin
		]);
		expect(selectTailMessages([u1, call, tool1, tool2, fin], 0, 1e6)).toEqual([]);
	});

	it('drops from the front to fit the budget, skipping orphaned tool results', () => {
		const big = msg(MessageRole.TOOL, 'z'.repeat(3000));

		expect(selectTailMessages([u1, call, big, fin], 4, 100)).toEqual([fin]);
	});
});

describe('W1: compaction against the cached prompt', () => {
	afterEach(() => {
		vi.restoreAllMocks();
		vi.unstubAllGlobals();
		conversationsStore.activeConversation = null;
		conversationsStore.activeMessages = [];
		chatStore.setChatLoading('c1', false);
	});

	function sse(content: string, timings?: object): Response {
		const chunks = [
			`data: ${JSON.stringify({ choices: [{ delta: { content } }] })}\n\n`,
			timings ? `data: ${JSON.stringify({ choices: [{ delta: {} }], timings })}\n\n` : '',
			'data: [DONE]\n\n'
		];

		return new Response(chunks.join(''), { headers: { 'Content-Type': 'text/event-stream' } });
	}

	async function scenario(full: boolean) {
		const N_CTX = 20000;
		const conv = {
			currNode: 'a2',
			id: 'c1',
			lastModified: 0,
			name: 'Plan'
		} as DatabaseConversation;
		const rows: DatabaseMessage[] = [
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
		const db = new Map<string, DatabaseMessage>(
			[
				{ ...msg(MessageRole.SYSTEM, ''), id: 'root', type: 'root' } as DatabaseMessage,
				...rows
			].map((m) => [m.id, structuredClone(m)])
		);
		const convs = new Map<string, DatabaseConversation>([[conv.id, structuredClone(conv)]]);

		let nextId = 0;

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
		vi.spyOn(DatabaseService, 'updateMessage').mockResolvedValue();
		vi.spyOn(DatabaseService, 'updateCurrentNode').mockResolvedValue();
		vi.spyOn(conversationsStore, 'loadConversations').mockResolvedValue();
		vi.spyOn(conversationsStore, 'loadConversation').mockResolvedValue(true);
		vi.spyOn(modelsStore.props, 'getModelContextSize').mockReturnValue(N_CTX);
		vi.spyOn(modelsStore, 'activeModelId', 'get').mockReturnValue('m');
		vi.spyOn(agenticStore, 'getConfig').mockReturnValue({ enabled: false } as never);
		vi.spyOn(agenticStore, 'runAgenticFlow').mockResolvedValue({ handled: false });

		const bodies: Record<string, unknown>[] = [];

		vi.stubGlobal(
			'fetch',
			vi.fn(async (_url: string, init?: RequestInit) => {
				bodies.push(JSON.parse(String(init?.body)));

				return sse(bodies.length === 1 ? 'SUMMARY TEXT' : 'reply', { predicted_n: 3 });
			})
		);

		settingsStore.config = {
			...settingsStore.config,
			compactKeepMessages: 2,
			contextReserveTokens: 2000,
			customJson: '{"tag":"w1"}'
		};
		conversationsStore.activeConversation = structuredClone(conv);
		conversationsStore.activeMessages = rows.map((m) => structuredClone(m));

		const result = await chatStore.compactConversation('c1');

		expect(result).toEqual({ convId: 'conv-1' });

		// the next real turn of the old conversation, as the chat store sends it
		const userNew = { ...msg(MessageRole.USER, 'question three'), id: 'u3', parent: 'a2' };
		const assistantNew = { ...msg(MessageRole.ASSISTANT, ''), id: 'a3', parent: 'u3' };

		await chatStore.streamChatCompletion([...rows, userNew], assistantNew);

		return { bodies, convs, db };
	}

	it('the summary request = the next turn with an instruction instead of the user message', async () => {
		const { bodies, convs, db } = await scenario(false);
		const [summary, next] = bodies;
		const sumMessages = summary.messages as Record<string, unknown>[];
		const nextMessages = next.messages as Record<string, unknown>[];
		const rest = (b: Record<string, unknown>) => {
			const r = { ...b };

			delete r.messages;
			delete r.max_tokens;
			delete r.thinking_budget_tokens;

			return r;
		};

		expect(sumMessages.slice(0, -1)).toEqual(nextMessages.slice(0, -1));
		expect(sumMessages.at(-1)).toMatchObject({ role: 'user' });
		expect(String(sumMessages.at(-1)!.content)).toContain('continuation summary');
		expect(rest(summary)).toEqual(rest(next));
		expect(summary).toMatchObject({ stream: true, tag: 'w1' });
		// max_tokens <= reserve - instruction; thinking capped at a quarter of it
		expect(summary.max_tokens as number).toBeLessThanOrEqual(2000);
		expect(summary.thinking_budget_tokens).toBe(Math.floor((summary.max_tokens as number) / 4));
		// the normal reply is capped at free - reserve
		const used = 1000 + 9000 + 20 + Math.ceil('question three'.length / 3) + 8;

		expect(next.max_tokens).toBe(20000 - used - 2000);

		// the new conversation: system, summary note, last 2 messages; linked, the old one kept
		const newConv = convs.get('conv-1')!;

		expect(newConv).toMatchObject({ compactedFromConversationId: 'c1', name: 'Plan (cont.)' });
		expect(convs.get('c1')).toMatchObject({ compactedIntoConversationId: 'conv-1' });

		const path: DatabaseMessage[] = [];

		for (let id: string | null = newConv.currNode; id; ) {
			const m = db.get(id)!;

			path.unshift(m);
			id = m.parent;
		}

		expect(path.map((m) => [m.role, m.content.slice(0, 20)])).toEqual([
			['system', ''],
			['system', 'be brief'],
			['assistant', 'Summary of the earli'],
			['user', 'question two'],
			['assistant', 'answer two']
		]);
		expect(path[2].content).toContain('SUMMARY TEXT');
		// old measurements do not travel
		expect(path[4].timings).toBeUndefined();
		expect(db.get('a2')).toBeDefined();
	}, 60000);

	it('a full context drops the truncated reply from the summary request', async () => {
		const { bodies } = await scenario(true);
		const sumMessages = bodies[0].messages as Record<string, unknown>[];

		expect(sumMessages.map((m) => m.role)).toEqual(['system', 'user', 'assistant', 'user', 'user']);
		expect(sumMessages[3]).toMatchObject({ content: 'question two' });
		// room = n_ctx - (prompt before the truncated reply) - instruction - 1, capped at the reserve
		expect(bodies[0].max_tokens as number).toBeLessThanOrEqual(2000);
	}, 60000);
});

describe('W1: later turns of an agentic flow', () => {
	afterEach(() => vi.restoreAllMocks());

	it('cap at free - reserve after the tool results; 0 = stop the flow', () => {
		vi.spyOn(modelsStore.props, 'getModelContextSize').mockReturnValue(10000);
		settingsStore.config = { ...settingsStore.config, contextReserveTokens: 2000, max_tokens: -1 };

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

		// 5000 measured + ~108 for the tool result
		expect(cap(t(100, 4800, 100), turn, 'm')).toBe(10000 - 5000 - 108 - 2000);
		expect(cap(t(100, 7500, 100), turn, 'm')).toBe(0);
		settingsStore.config = { ...settingsStore.config, contextReserveTokens: 0 };
		expect(cap(t(100, 7500, 100), turn, 'm')).toBeUndefined();
	});
});
