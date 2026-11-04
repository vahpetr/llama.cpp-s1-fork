import { MessageRole, MessageType } from '$lib/enums';
import type { DatabaseMessage } from '$lib/types/database';
import { agenticCopyTarget } from '$lib/utils/agentic';
import { describe, expect, it } from 'vitest';

let nextId = 0;

function makeMsg(role: MessageRole, opts: Partial<DatabaseMessage> = {}): DatabaseMessage {
	nextId++;

	return {
		children: [],
		content: '',
		convId: 'conv-1',
		id: `msg-${nextId}`,
		parent: null,
		role,
		timestamp: nextId,
		type: MessageType.TEXT,
		...opts
	};
}

const toolCalls = JSON.stringify([{ function: { arguments: '{}', name: 'ls' }, id: 'c1' }]);

describe('agenticCopyTarget', () => {
	it('plain reply copies its own content', () => {
		const m = makeMsg(MessageRole.ASSISTANT, { content: 'hello' });

		expect(agenticCopyTarget(m).content).toBe('hello');
	});

	it('agentic entry with empty root copies the final assistant', () => {
		const root = makeMsg(MessageRole.ASSISTANT, { reasoningContent: 'think', toolCalls });
		const tool = makeMsg(MessageRole.TOOL, { content: 'a.txt', toolCallId: 'c1' });
		const mid = makeMsg(MessageRole.ASSISTANT, { content: 'Looking further.', toolCalls });
		const tool2 = makeMsg(MessageRole.TOOL, { content: 'b.txt', toolCallId: 'c1' });
		const final = makeMsg(MessageRole.ASSISTANT, { content: 'Final answer.' });

		expect(agenticCopyTarget(root, [tool, mid, tool2, final]).content).toBe('Final answer.');
	});

	it('empty final assistant falls back to all assistant step texts', () => {
		const root = makeMsg(MessageRole.ASSISTANT, { content: 'Step one.', toolCalls });
		const tool = makeMsg(MessageRole.TOOL, { content: 'x', toolCallId: 'c1' });
		const mid = makeMsg(MessageRole.ASSISTANT, { content: ' Step two. ', toolCalls });
		const tool2 = makeMsg(MessageRole.TOOL, { content: 'y', toolCallId: 'c1' });
		const final = makeMsg(MessageRole.ASSISTANT, { content: '  ' });

		expect(agenticCopyTarget(root, [tool, mid, tool2, final]).content).toBe(
			'Step one.\n\nStep two.'
		);
	});

	it('root with tool results but no later assistant copies the root', () => {
		const root = makeMsg(MessageRole.ASSISTANT, { content: 'Calling.', toolCalls });
		const tool = makeMsg(MessageRole.TOOL, { content: 'x', toolCallId: 'c1' });

		expect(agenticCopyTarget(root, [tool]).content).toBe('Calling.');
	});

	it('final extras travel with the final content', () => {
		const extra = [{ content: 'f', name: 'n.txt', type: 'TEXT' }] as never;
		const root = makeMsg(MessageRole.ASSISTANT, { toolCalls });
		const tool = makeMsg(MessageRole.TOOL, { content: 'x', toolCallId: 'c1' });
		const final = makeMsg(MessageRole.ASSISTANT, { content: 'done', extra });

		expect(agenticCopyTarget(root, [tool, final]).extra).toBe(extra);
	});
});
