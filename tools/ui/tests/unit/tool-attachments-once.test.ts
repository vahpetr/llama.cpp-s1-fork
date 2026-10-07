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

import {
	AttachmentType,
	BuiltInTool,
	MessageRole,
	ToolPermissionDecision,
	ToolSource
} from '$lib/enums';
import { ChatService } from '$lib/services/chat.service';
import { DatabaseService } from '$lib/services/database.service';
import { ToolsService } from '$lib/services/tools.service';
import { agenticStore } from '$lib/stores/agentic/index.svelte';
import { chatStore } from '$lib/stores/chat/index.svelte';
import { conversationsStore } from '$lib/stores/conversations/index.svelte';
import { mcpStore } from '$lib/stores/mcp/index.svelte';
import { toolsStore } from '$lib/stores/tools.svelte';
import type {
	DatabaseConversation,
	DatabaseMessage,
	DatabaseMessageExtra,
	DatabaseMessageExtraImageFile
} from '$lib/types/database';

const IMG = 'data:image/png;base64,iVBORw0KGgo=';
const IMG2 = 'data:image/png;base64,R0lGODlh';

function msg(overrides: Partial<DatabaseMessage>): DatabaseMessage {
	return {
		children: [],
		content: '',
		convId: 'conv-w1',
		id: 'm',
		parent: null,
		role: MessageRole.ASSISTANT,
		timestamp: 0,
		toolCalls: '',
		type: 'text',
		...overrides
	} as DatabaseMessage;
}

// in-memory db + the real chat store callbacks driving the real agentic loop: one tool call
// whose result carries one image, then a final text turn
async function runFlowWithToolImage(streaming: boolean): Promise<DatabaseMessage> {
	const db = new Map<string, DatabaseMessage>();

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
	vi.spyOn(DatabaseService, 'updateCurrentNode').mockImplementation(async () => {});

	const assistant = msg({ id: 'ast-1' });

	db.set(assistant.id, structuredClone(assistant));
	conversationsStore.activeConversation = {
		currNode: assistant.id,
		id: 'conv-w1',
		lastModified: 0,
		name: 'w1'
	} as DatabaseConversation;
	conversationsStore.activeMessages = [assistant];

	const toolName = streaming ? BuiltInTool.SERVER_EXEC_SHELL_COMMAND : 'shot';

	vi.spyOn(toolsStore, 'getToolSource').mockReturnValue(
		streaming ? ToolSource.SERVER : ToolSource.MCP
	);
	vi.spyOn(toolsStore, 'getToolServerLabel').mockReturnValue('srv');
	vi.spyOn(
		(agenticStore as unknown as { gates: { requestPermission: () => Promise<string> } }).gates,
		'requestPermission'
	).mockResolvedValue(ToolPermissionDecision.ONCE);
	vi.spyOn(mcpStore, 'executeTool').mockResolvedValue({ content: `shot taken\n${IMG}` } as never);
	vi.spyOn(ToolsService, 'streamTool').mockImplementation(async function* () {
		yield { chunk: `shot taken\n${IMG}`, done: false } as never;
		yield { chunk: null, done: true } as never;
	});

	let turn = 0;

	vi.spyOn(ChatService, 'sendMessage').mockImplementation(async (_messages, options) => {
		if (turn++ === 0) {
			options?.onToolCallChunk?.(
				JSON.stringify([
					{ function: { arguments: '{}', name: toolName }, id: 'call_1', type: 'function' }
				])
			);
		} else {
			options?.onChunk?.('done');
		}
	});

	// run the real loop with the callbacks the chat store builds
	vi.spyOn(agenticStore, 'runAgenticFlow').mockImplementation(async (params) => {
		await (
			agenticStore as unknown as { executeAgenticLoop: (p: unknown) => Promise<void> }
		).executeAgenticLoop({
			agenticConfig: { enabled: true, maxTurns: 5 },
			callbacks: params.callbacks,
			conversationId: params.conversationId,
			messages: [],
			options: {},
			signal: params.signal,
			tools: [{ function: { name: toolName, parameters: {} }, type: 'function' }]
		});

		return { handled: true };
	});

	await chatStore.streamChatCompletion([assistant], assistant);
	// let fire-and-forget db writes settle
	await new Promise((r) => setTimeout(r, 0));

	const toolRows = [...db.values()].filter((m) => m.role === MessageRole.TOOL);

	expect(toolRows).toHaveLength(1);

	return toolRows[0];
}

describe('W1: tool-result attachments are stored once', () => {
	afterEach(() => {
		vi.restoreAllMocks();
		conversationsStore.activeConversation = null;
		conversationsStore.activeMessages = [];
	});

	it('create path: the db row holds exactly one copy of the image', async () => {
		const row = await runFlowWithToolImage(false);

		expect(row.extra).toHaveLength(1);
		expect((row.extra![0] as DatabaseMessageExtraImageFile).base64Url).toBe(IMG);
		const active = conversationsStore.activeMessages.find((m) => m.id === row.id);

		expect(active?.extra).toHaveLength(1);
	}, 60000);

	it('streaming update path: the db row holds exactly one copy of the image', async () => {
		const row = await runFlowWithToolImage(true);

		expect(row.extra).toHaveLength(1);
		expect((row.extra![0] as DatabaseMessageExtraImageFile).base64Url).toBe(IMG);
	}, 60000);
});

describe('W1: the history rebuild repairs doubled legacy lists', () => {
	const img = (url: string, name = 'mcp-attachment-1.png'): DatabaseMessageExtra =>
		({ base64Url: url, name, type: AttachmentType.IMAGE }) as DatabaseMessageExtra;
	const toolMsg = (extra: DatabaseMessageExtra[]) =>
		msg({ content: 'shot taken', extra, role: MessageRole.TOOL, toolCallId: 'call_1' });
	const imageUrls = (content: unknown): string[] =>
		(content as { image_url?: { url: string } }[])
			.filter((p) => p.image_url)
			.map((p) => p.image_url!.url);

	it('a doubled list [a, a] is sent as one copy', async () => {
		const out = await ChatService.convertDbMessageToApiChatMessageData(
			toolMsg([img(IMG), img(IMG)])
		);

		expect(imageUrls(out.content)).toEqual([IMG]);
	});

	it('a doubled list [a, b, a, b] is sent as [a, b]', async () => {
		const out = await ChatService.convertDbMessageToApiChatMessageData(
			toolMsg([img(IMG), img(IMG2), img(IMG), img(IMG2)])
		);

		expect(imageUrls(out.content)).toEqual([IMG, IMG2]);
	});

	it('a repeated attachment that is not a doubled list is kept', async () => {
		const out = await ChatService.convertDbMessageToApiChatMessageData(
			toolMsg([img(IMG), img(IMG2), img(IMG2)])
		);

		expect(imageUrls(out.content)).toEqual([IMG, IMG2, IMG2]);

		const halvesDiffer = await ChatService.convertDbMessageToApiChatMessageData(
			toolMsg([img(IMG), img(IMG2), img(IMG2), img(IMG)])
		);

		expect(imageUrls(halvesDiffer.content)).toEqual([IMG, IMG2, IMG2, IMG]);
	});
});
