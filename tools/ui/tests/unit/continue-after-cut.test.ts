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

import { MessageRole, ToolPermissionDecision, ToolSource } from '$lib/enums';
import { ChatService } from '$lib/services/chat.service';
import { DatabaseService } from '$lib/services/database.service';
import { ToolsService } from '$lib/services/tools.service';
import { agenticStore } from '$lib/stores/agentic/index.svelte';
import { chatStore } from '$lib/stores/chat/index.svelte';
import { conversationsStore } from '$lib/stores/conversations/index.svelte';
import { toolsStore } from '$lib/stores/tools.svelte';
import type { ApiChatMessageData } from '$lib/types/api';
import type { DatabaseConversation, DatabaseMessage } from '$lib/types/database';

/**
 * W1 (2026-11-07, chat a4643898): a reply cut by a server restart, then Continue. The model
 * answered the continuation with a tool call; the manual Continue path kept only text, so the
 * call was dropped and the reply showed «Reasoning / Cancelled». A Continue in a conversation
 * with tools now runs as the first turn of the agentic flow.
 */

const TOOLS = [
	{ function: { name: 'web_search', parameters: { type: 'object' } }, type: 'function' }
];

function msg(overrides: Partial<DatabaseMessage>): DatabaseMessage {
	return {
		children: [],
		content: '',
		convId: 'conv-cut',
		id: 'm',
		parent: null,
		role: MessageRole.ASSISTANT,
		timestamp: 0,
		toolCalls: '',
		type: 'text',
		...overrides
	} as DatabaseMessage;
}

function chain(partial: Partial<DatabaseMessage>): DatabaseMessage[] {
	return [
		msg({ content: 'find the link', id: 'u1', role: MessageRole.USER }),
		msg({
			id: 'a1',
			parent: 'u1',
			reasoningContent: 'search',
			toolCalls: JSON.stringify([
				{ function: { arguments: '{}', name: 'web_search' }, id: 'c1', type: 'function' }
			])
		}),
		msg({ content: 'RESULT', id: 't1', parent: 'a1', role: MessageRole.TOOL, toolCallId: 'c1' }),
		msg({ id: 'a2', parent: 't1', ...partial })
	];
}

function setConversation(messages: DatabaseMessage[], tools: unknown[] = TOOLS) {
	const conv = {
		currNode: messages[messages.length - 1].id,
		id: 'conv-cut',
		lastModified: 0,
		name: 'cut',
		toolsSnapshot: JSON.stringify(tools)
	} as DatabaseConversation;

	conversationsStore.activeConversation = conv;
	conversationsStore.activeMessages = messages;
	vi.spyOn(DatabaseService, 'getConversation').mockResolvedValue(conv);
	vi.spyOn(DatabaseService, 'getConversationMessages').mockResolvedValue(messages);
	vi.spyOn(agenticStore, 'getConfig').mockReturnValue({ enabled: true, maxTurns: 5 } as never);
}

afterEach(() => {
	vi.restoreAllMocks();
	// a mocked stream never ends the loading state the flow set
	chatStore.setChatLoading('conv-cut', false);
	conversationsStore.activeConversation = null;
	conversationsStore.activeMessages = [];
});

describe('W1: Continue after a cut, routing', () => {
	it('a reply cut before its first token runs its turn fresh into the same message', async () => {
		const messages = chain({ content: '' });

		setConversation(messages);
		const stream = vi.spyOn(chatStore, 'streamChatCompletion').mockResolvedValue();
		const send = vi.spyOn(ChatService, 'sendMessage');

		await chatStore.continueAssistantMessage('a2');

		expect(send).not.toHaveBeenCalled();
		expect(stream).toHaveBeenCalledTimes(1);
		const [path, target, , , , , resume] = stream.mock.calls[0];

		expect(path.map((m) => m.id)).toEqual(['u1', 'a1', 't1']);
		expect(target.id).toBe('a2');
		expect(resume).toEqual({ continue: false, message: expect.objectContaining({ id: 'a2' }) });
	});

	it('a whitespace-only partial is continued, not regenerated (literal emptiness)', async () => {
		const messages = chain({ content: '\n' });

		setConversation(messages);
		const stream = vi.spyOn(chatStore, 'streamChatCompletion').mockResolvedValue();

		await chatStore.continueAssistantMessage('a2');

		const [path, , , , , , resume] = stream.mock.calls[0];

		expect(path.map((m) => m.id)).toEqual(['u1', 'a1', 't1', 'a2']);
		expect(resume?.continue).toBe(true);
	});

	it('a partial reply continues as the first turn of the agentic flow', async () => {
		const messages = chain({ reasoningContent: 'Correct links:\n' });

		setConversation(messages);
		const stream = vi.spyOn(chatStore, 'streamChatCompletion').mockResolvedValue();

		await chatStore.continueAssistantMessage('a2');

		expect(stream).toHaveBeenCalledTimes(1);
		const [path, target, , , , , resume] = stream.mock.calls[0];

		expect(path.map((m) => m.id)).toEqual(['u1', 'a1', 't1', 'a2']);
		expect(target.id).toBe('a2');
		expect(resume).toEqual({ continue: true, message: expect.objectContaining({ id: 'a2' }) });
	});

	it('a conversation whose frozen tool list is empty keeps the plain continuation', async () => {
		const messages = chain({ reasoningContent: 'Correct links:\n' });

		// the first request fixes the list, also an empty one (every tool disabled at that send)
		setConversation(messages, []);
		vi.mocked(agenticStore.getConfig).mockRestore();
		const stream = vi.spyOn(chatStore, 'streamChatCompletion').mockResolvedValue();
		const send = vi.spyOn(ChatService, 'sendMessage').mockResolvedValue(undefined as never);

		await chatStore.continueAssistantMessage('a2');

		expect(stream).not.toHaveBeenCalled();
		expect(send).toHaveBeenCalledTimes(1);
		expect(send.mock.calls[0][1]?.continueFinalMessage).toBe(true);
	});
});

describe('W1: a continued first turn runs the tool call it asks for', () => {
	it('joins the echo, replaces the partial in the history and runs the call', async () => {
		const messages = chain({ reasoningContent: 'Correct links:\n' });
		const db = new Map<string, DatabaseMessage>(messages.map((m) => [m.id, structuredClone(m)]));

		let nextId = 0;

		setConversation(messages);
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
		vi.spyOn(DatabaseService, 'updateCurrentNode').mockImplementation(async () => {});
		vi.spyOn(toolsStore, 'getToolSource').mockReturnValue(ToolSource.SERVER);
		vi.spyOn(toolsStore, 'getToolServerLabel').mockReturnValue('srv');
		vi.spyOn(
			(agenticStore as unknown as { gates: { requestPermission: () => Promise<string> } }).gates,
			'requestPermission'
		).mockResolvedValue(ToolPermissionDecision.ONCE);
		vi.spyOn(ToolsService, 'executeTool').mockResolvedValue({
			content: 'README updated',
			isError: false
		} as never);

		const requests: { continued: unknown; messages: ApiChatMessageData[] }[] = [];

		vi.spyOn(ChatService, 'sendMessage').mockImplementation(async (sent, options) => {
			requests.push({
				continued: options?.continueFinalMessage,
				messages: structuredClone(sent as ApiChatMessageData[])
			});

			if (requests.length === 1) {
				// llama-server streams the prefilled partial back first, then the new tokens
				options?.onReasoningChunk?.('Correct links:\n');
				options?.onReasoningChunk?.('update README.\n');
				options?.onToolCallChunk?.(
					JSON.stringify([
						{ function: { arguments: '{}', name: 'web_search' }, id: 'c2', type: 'function' }
					])
				);
			} else {
				options?.onChunk?.('done');
			}
		});
		// the real loop with the chat store's callbacks (the tool snapshot comes from the conversation)
		vi.spyOn(agenticStore, 'runAgenticFlow').mockImplementation(async (params) => {
			await (
				agenticStore as unknown as { executeAgenticLoop: (p: unknown) => Promise<void> }
			).executeAgenticLoop({
				agenticConfig: { enabled: true, maxTurns: 5 },
				callbacks: params.callbacks,
				continuation: params.continuation,
				conversationId: params.conversationId,
				messages: await ChatService.normalizeMessagesForApi(params.messages),
				options: params.options ?? {},
				signal: params.signal,
				tools: TOOLS
			});

			return { handled: true };
		});

		const partial = messages[3];

		await chatStore.streamChatCompletion(
			messages,
			partial,
			undefined,
			undefined,
			undefined,
			undefined,
			{
				continue: true,
				message: partial
			}
		);
		await new Promise((r) => setTimeout(r, 0));

		// turn 1 continues the partial, turn 2 is a fresh turn after the tool result
		expect(requests).toHaveLength(2);
		expect(requests[0].continued).toBe(true);
		expect(requests[0].messages.at(-1)).toMatchObject({
			reasoning_content: 'Correct links:\n',
			role: MessageRole.ASSISTANT
		});
		expect(requests[1].continued).toBeUndefined();
		const history = requests[1].messages.slice(3);

		expect(history.map((m) => m.role)).toEqual([MessageRole.ASSISTANT, MessageRole.TOOL]);
		expect(history[0].reasoning_content).toBe('Correct links:\nupdate README.\n');
		expect(history[0].tool_calls?.[0]?.id).toBe('c2');
		expect(history[1]).toMatchObject({ content: 'README updated', tool_call_id: 'c2' });

		// the partial row holds the joined reasoning + the call; the tool ran; the flow answered
		const row = db.get('a2')!;

		expect(row.reasoningContent).toBe('Correct links:\nupdate README.\n');
		expect(JSON.parse(row.toolCalls!)[0].id).toBe('c2');
		const created = [...db.values()].filter((m) => m.id.startsWith('db-'));

		expect(created.map((m) => m.role)).toEqual([MessageRole.TOOL, MessageRole.ASSISTANT]);
		expect(created[1].content).toBe('done');
	}, 60000);
});

describe('W1: a resumed reply keeps its request settings and the echo check', () => {
	function captureFlow() {
		const updates: Record<string, unknown>[] = [];

		vi.spyOn(DatabaseService, 'updateMessage').mockImplementation(async (_id, u) => {
			updates.push(u as Record<string, unknown>);
		});
		vi.spyOn(DatabaseService, 'updateCurrentNode').mockImplementation(async () => {});

		const sent: Record<string, unknown>[] = [];

		vi.spyOn(ChatService, 'sendMessage').mockImplementation(async (_m, options) => {
			sent.push(options as Record<string, unknown>);
			// the stream ends inside the echo of the partial: nothing was appended
			options?.onReasoningChunk?.('Correct links: the partial');
		});
		vi.spyOn(agenticStore, 'runAgenticFlow').mockImplementation(async (params) => {
			await (
				agenticStore as unknown as { executeAgenticLoop: (p: unknown) => Promise<void> }
			).executeAgenticLoop({
				agenticConfig: { enabled: true, maxTurns: 5 },
				callbacks: params.callbacks,
				continuation: params.continuation,
				conversationId: params.conversationId,
				messages: await ChatService.normalizeMessagesForApi(params.messages),
				options: params.options ?? {},
				signal: params.signal,
				tools: TOOLS
			});

			return { handled: true };
		});

		return { sent, updates };
	}

	it('a restarted empty reply runs with its saved requestOptions, not the current settings', async () => {
		const messages = chain({
			content: '',
			requestOptions: { reasoningEffortNative: 'low', sampling: { temperature: 0.123 } }
		});

		setConversation(messages);
		vi.spyOn(chatStore, 'getApiOptions').mockReturnValue({
			reasoningEffortNative: 'high',
			temperature: 0.9
		});
		const { sent } = captureFlow();

		await chatStore.streamChatCompletion(
			messages.slice(0, 3),
			messages[3],
			undefined,
			undefined,
			undefined,
			undefined,
			{ continue: false, message: messages[3] }
		);

		expect(sent).toHaveLength(1);
		expect(sent[0].temperature).toBe(0.123);
		expect(sent[0].reasoningEffortNative).toBe('low');
		expect(sent[0].continueFinalMessage).toBeUndefined();
	}, 60000);

	it('a stream that ended inside the echo is persisted as echoMismatch, the partial kept', async () => {
		const messages = chain({ reasoningContent: 'Correct links: the partial reasoning so far\n' });

		setConversation(messages);
		const { sent, updates } = captureFlow();

		await chatStore.streamChatCompletion(
			messages,
			messages[3],
			undefined,
			undefined,
			undefined,
			undefined,
			{ continue: true, message: messages[3] }
		);

		expect(sent[0].continueFinalMessage).toBe(true);
		const final = updates.find((u) => 'reasoningContent' in u);

		expect(final?.echoMismatch).toBe(true);
		expect(final?.reasoningContent).toBe('Correct links: the partial reasoning so far\n');
	}, 60000);
});

describe('W1: a manual Continue keeps the plain Continue semantics', () => {
	function runFlow(lastTurn: Record<string, unknown>) {
		vi.spyOn(DatabaseService, 'updateMessage').mockImplementation(async () => {});
		vi.spyOn(DatabaseService, 'updateCurrentNode').mockImplementation(async () => {});
		vi.spyOn(agenticStore, 'runAgenticFlow').mockResolvedValue({ handled: true });
		vi.spyOn(agenticStore, 'takeLastTurn').mockReturnValue(lastTurn as never);

		return vi
			.spyOn(chatStore as unknown as { afterReply: () => Promise<void> }, 'afterReply')
			.mockResolvedValue();
	}

	it('no auto-continuation (afterReply) after a resumed reply', async () => {
		const messages = chain({ reasoningContent: 'Correct links:\n' });

		setConversation(messages);
		const afterReply = runFlow({ finish: 'length', stopType: 'limit' });
		const take = vi.mocked(agenticStore.takeLastTurn);

		await chatStore.streamChatCompletion(
			messages,
			messages[3],
			undefined,
			undefined,
			undefined,
			undefined,
			{ continue: true, message: messages[3] }
		);

		expect(take).toHaveBeenCalledTimes(1);
		expect(afterReply).not.toHaveBeenCalled();
	});

	it('a normal flow still runs afterReply', async () => {
		const messages = chain({ content: '' });

		setConversation(messages);
		const afterReply = runFlow({ finish: 'length', stopType: 'limit' });

		await chatStore.streamChatCompletion(messages.slice(0, 3), messages[3]);

		expect(afterReply).toHaveBeenCalledTimes(1);
	});

	it('a resume the agentic flow did not handle surfaces an error and sends nothing', async () => {
		const messages = chain({ reasoningContent: 'Correct links:\n' });

		setConversation(messages);
		vi.spyOn(agenticStore, 'runAgenticFlow').mockResolvedValue({ handled: false });
		const send = vi.spyOn(ChatService, 'sendMessage');
		const dialog = vi.spyOn(chatStore, 'showErrorDialog');

		await chatStore.streamChatCompletion(
			messages,
			messages[3],
			undefined,
			undefined,
			undefined,
			undefined,
			{ continue: true, message: messages[3] }
		);

		expect(send).not.toHaveBeenCalled();
		expect(dialog).toHaveBeenCalledWith(expect.objectContaining({ message: expect.any(String) }));
	});
});
