import { MessageRole } from '$lib/enums';
import { filterByLeafNodeId } from '$lib/utils/branching';
import { describe, expect, it } from 'vitest';

const msg = (
	id: string,
	parent: string | null,
	role: string,
	timestamp: number,
	extra: Partial<DatabaseMessage> = {}
): DatabaseMessage =>
	({
		children: [],
		content: id,
		convId: 'c',
		id,
		parent,
		role,
		timestamp,
		toolCalls: '',
		type: parent === null ? 'root' : 'text',
		...extra
	}) as DatabaseMessage;
const calls = (...ids: string[]) =>
	JSON.stringify(
		ids.map((id) => ({ function: { arguments: '{}', name: 'f' }, id, type: 'function' }))
	);

describe('filterByLeafNodeId and parallel tool results', () => {
	it('keeps every result of a legacy parallel call (results stored as siblings)', () => {
		const messages = [
			msg('root', null, MessageRole.SYSTEM, 0),
			msg('u1', 'root', MessageRole.USER, 1),
			msg('a1', 'u1', MessageRole.ASSISTANT, 2, { toolCalls: calls('cA', 'cB') }),
			msg('tA', 'a1', MessageRole.TOOL, 3, { toolCallId: 'cA' }),
			msg('tB', 'a1', MessageRole.TOOL, 4, { toolCallId: 'cB' }),
			msg('a2', 'tB', MessageRole.ASSISTANT, 5),
			msg('u2', 'a2', MessageRole.USER, 6)
		];
		const path = filterByLeafNodeId(messages, 'u2').map((m) => m.id);

		expect(path).toEqual(['u1', 'a1', 'tA', 'tB', 'a2', 'u2']);
	});

	it('orders legacy siblings by tool-call order, not by which one is on the path', () => {
		const messages = [
			msg('root', null, MessageRole.SYSTEM, 0),
			msg('u1', 'root', MessageRole.USER, 1),
			msg('a1', 'u1', MessageRole.ASSISTANT, 2, { toolCalls: calls('cA', 'cB') }),
			msg('tB', 'a1', MessageRole.TOOL, 3, { toolCallId: 'cB' }),
			msg('tA', 'a1', MessageRole.TOOL, 3, { toolCallId: 'cA' }),
			msg('a2', 'tA', MessageRole.ASSISTANT, 4)
		];

		expect(filterByLeafNodeId(messages, 'a2').map((m) => m.id)).toEqual([
			'u1',
			'a1',
			'tA',
			'tB',
			'a2'
		]);
	});

	it('follows tree order for chained results that share a timestamp', () => {
		const messages = [
			msg('root', null, MessageRole.SYSTEM, 0),
			msg('u1', 'root', MessageRole.USER, 1),
			msg('a1', 'u1', MessageRole.ASSISTANT, 2, { toolCalls: calls('cA', 'cB') }),
			msg('tA', 'a1', MessageRole.TOOL, 3, { toolCallId: 'cA' }),
			msg('tB', 'tA', MessageRole.TOOL, 3, { toolCallId: 'cB' }),
			msg('a2', 'tB', MessageRole.ASSISTANT, 3)
		];

		expect(filterByLeafNodeId(messages, 'a2').map((m) => m.id)).toEqual([
			'u1',
			'a1',
			'tA',
			'tB',
			'a2'
		]);
	});

	it('keeps a system message first and leaves real branches out', () => {
		const messages = [
			msg('root', null, MessageRole.SYSTEM, 0),
			msg('s', 'root', MessageRole.SYSTEM, 5),
			msg('u1', 's', MessageRole.USER, 1),
			msg('a1', 'u1', MessageRole.ASSISTANT, 2),
			msg('a1b', 'u1', MessageRole.ASSISTANT, 3)
		];

		expect(filterByLeafNodeId(messages, 'a1b').map((m) => m.id)).toEqual(['s', 'u1', 'a1b']);
	});
});
