import { afterEach, beforeAll, describe, expect, it, vi } from 'vitest';

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

import { MessageRole, ReasoningEffort, ToolPermissionDecision, ToolSource } from '$lib/enums';
import { ChatService } from '$lib/services/chat.service';
import { DatabaseService } from '$lib/services/database.service';
import { agenticStore } from '$lib/stores/agentic/index.svelte';
import { chatStore } from '$lib/stores/chat/index.svelte';
import { conversationsStore } from '$lib/stores/conversations/index.svelte';
import { mcpStore } from '$lib/stores/mcp/index.svelte';
import { modelsStore } from '$lib/stores/models/index.svelte';
import { settingsStore } from '$lib/stores/settings/index.svelte';
import { toolsStore } from '$lib/stores/tools.svelte';
import type { DatabaseConversation, DatabaseMessage } from '$lib/types/database';
import { filterByLeafNodeId } from '$lib/utils';

const IMG = 'data:image/png;base64,iVBORw0KGgo=';
const TOOLS = [
	{
		function: { description: 'take a shot', name: 'shot', parameters: { type: 'object' } },
		type: 'function'
	}
];

function sse(chunks: object[]): Response {
	const body =
		chunks.map((c) => `data: ${JSON.stringify({ choices: [c] })}\n\n`).join('') +
		'data: [DONE]\n\n';

	return new Response(body, { headers: { 'Content-Type': 'text/event-stream' } });
}

describe('W2: pre-encode after a tool-call flow', () => {
	afterEach(() => {
		vi.restoreAllMocks();
		vi.unstubAllGlobals();
		conversationsStore.activeConversation = null;
		conversationsStore.activeMessages = [];
	});

	it('posts the conversation the next request rebuilds, through the same request builder', async () => {
		const db = new Map<string, DatabaseMessage>();
		const conv = {
			currNode: 'ast-1',
			id: 'conv-w2',
			lastModified: 0,
			name: 'w2',
			reasoningEffort: ReasoningEffort.HIGH,
			toolsSnapshot: JSON.stringify(TOOLS)
		} as unknown as DatabaseConversation;

		let nextId = 0;

		vi.spyOn(DatabaseService, 'createMessageBranch').mockImplementation(async (m, parent) => {
			const created = {
				...(m as DatabaseMessage),
				children: [],
				id: `db-${++nextId}`,
				parent
			} as DatabaseMessage;

			db.set(created.id, structuredClone(created));

			return created;
		});
		vi.spyOn(DatabaseService, 'updateMessage').mockImplementation(async (id, updates) => {
			const row = db.get(id);

			if (row) db.set(id, structuredClone({ ...row, ...updates }));
		});
		vi.spyOn(DatabaseService, 'updateCurrentNode').mockImplementation(async (_c, node) => {
			conv.currNode = node;
		});
		vi.spyOn(DatabaseService, 'getConversation').mockImplementation(async () =>
			structuredClone(conv)
		);
		vi.spyOn(DatabaseService, 'getConversationMessages').mockImplementation(async () =>
			[...db.values()].map((m) => structuredClone(m))
		);

		const user = {
			children: [],
			content: 'take a shot',
			convId: 'conv-w2',
			id: 'usr-1',
			parent: null,
			role: MessageRole.USER,
			timestamp: 1,
			toolCalls: '',
			type: 'text'
		} as DatabaseMessage;
		const assistant = {
			...user,
			content: '',
			id: 'ast-1',
			parent: 'usr-1',
			role: MessageRole.ASSISTANT,
			timestamp: 2
		} as DatabaseMessage;

		db.set(user.id, structuredClone(user));
		db.set(assistant.id, structuredClone(assistant));
		conversationsStore.activeConversation = { ...conv };
		conversationsStore.activeMessages = [user, assistant];
		settingsStore.config = {
			...settingsStore.config,
			customJson: '{"cache_prompt_tag":"w2"}',
			preEncodeConversation: true
		};

		vi.spyOn(modelsStore.props, 'modelSupportsVision').mockReturnValue(true);
		vi.spyOn(toolsStore, 'getToolSource').mockReturnValue(ToolSource.MCP);
		vi.spyOn(toolsStore, 'getToolServerLabel').mockReturnValue('srv');
		vi.spyOn(
			(agenticStore as unknown as { gates: { requestPermission: () => Promise<string> } }).gates,
			'requestPermission'
		).mockResolvedValue(ToolPermissionDecision.ONCE);
		vi.spyOn(mcpStore, 'executeTool').mockResolvedValue({
			content: `shot taken\n${IMG}`
		} as never);
		vi.spyOn(agenticStore, 'runAgenticFlow').mockImplementation(async (params) => {
			await (
				agenticStore as unknown as { executeAgenticLoop: (p: unknown) => Promise<void> }
			).executeAgenticLoop({
				agenticConfig: { enabled: true, maxTurns: 5 },
				callbacks: params.callbacks,
				conversationId: params.conversationId,
				// as runAgenticFlow does
				messages: await ChatService.normalizeMessagesForApi(params.messages),
				options: params.options,
				signal: params.signal,
				tools: TOOLS
			});

			return { handled: true };
		});

		const bodies: Record<string, unknown>[] = [];

		let turn = 0;

		vi.stubGlobal(
			'fetch',
			vi.fn(async (url: string, init?: RequestInit) => {
				if (String(url).includes('slots')) return new Response('[]');

				bodies.push(JSON.parse(String(init?.body)));

				if (turn++ === 0) {
					return sse([
						{ delta: { reasoning_content: 'need a shot' } },
						{
							delta: {
								tool_calls: [
									{
										function: { arguments: '{}', name: 'shot' },
										id: 'call_1',
										index: 0,
										type: 'function'
									}
								]
							}
						}
					]);
				}

				return sse([{ delta: { content: 'done' } }]);
			})
		);

		await chatStore.streamChatCompletion([user], assistant);

		for (let i = 0; i < 100 && bodies.length < 3; i++) await new Promise((r) => setTimeout(r, 5));

		expect(bodies).toHaveLength(3);

		const [, lastTurn, preEncode] = bodies;
		const lastMessages = lastTurn.messages as Record<string, unknown>[];
		const preMessages = preEncode.messages as Record<string, unknown>[];

		// the last loop request had user, assistant(tool_calls, reasoning), tool(text + image)
		expect(lastMessages.map((m) => m.role)).toEqual(['user', 'assistant', 'tool']);
		expect(JSON.stringify(lastMessages[2].content)).toContain(IMG);
		expect(preEncode.n_predict).toBe(0);
		expect(preMessages.map((m) => m.role)).toEqual(['user', 'assistant', 'tool', 'assistant']);
		expect(preMessages[1]).toMatchObject({ reasoning_content: 'need a shot' });
		expect(preMessages[2]).toEqual(lastMessages[2]);
		expect(preMessages[3]).toMatchObject({ content: 'done', role: 'assistant' });

		// the next turn, as the chat store sends it: the path rebuilt from the db plus a new user
		const leaf = conv.currNode;
		const user2 = { ...user, content: 'again', id: 'usr-2', parent: leaf, timestamp: 3 };
		const assistant2 = { ...assistant, id: 'ast-2', parent: 'usr-2', timestamp: 4 };

		db.set(user2.id, structuredClone(user2));
		db.set(assistant2.id, structuredClone(assistant2));
		conv.currNode = 'ast-2';

		const next = filterByLeafNodeId([...db.values()], 'ast-2', false) as DatabaseMessage[];

		await chatStore.streamChatCompletion(next.slice(0, -1), assistant2);

		const nextTurn = bodies[3];
		const rest = (b: Record<string, unknown>) => {
			const r = { ...b };

			delete r.messages;
			delete r.n_predict;

			return r;
		};

		// pre-encode body = the next request's body minus the new user turn, n_predict aside
		expect(preMessages).toEqual((nextTurn.messages as unknown[]).slice(0, -1));
		expect(rest(preEncode)).toEqual(rest(nextTurn));
		expect(nextTurn).toMatchObject({
			cache_prompt_tag: 'w2',
			chat_template_kwargs: { enable_thinking: true },
			tools: TOOLS
		});
	}, 60000);
});
