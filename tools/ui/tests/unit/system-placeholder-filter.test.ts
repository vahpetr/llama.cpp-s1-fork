import { SYSTEM_MESSAGE_PLACEHOLDER } from '$lib/constants';
import { MessageRole } from '$lib/enums';
import { ChatService } from '$lib/services/chat.service';
import { describe, expect, it } from 'vitest';

describe('normalizeMessagesForApi and system messages', () => {
	it('drops the placeholder a system message is created with, like an empty one', async () => {
		const out = await ChatService.normalizeMessagesForApi([
			{ content: SYSTEM_MESSAGE_PLACEHOLDER, role: MessageRole.SYSTEM },
			{ content: '  ', role: MessageRole.SYSTEM },
			{ content: 'hi', role: MessageRole.USER }
		]);

		expect(out.map((m) => m.role)).toEqual([MessageRole.USER]);
	});

	it('keeps a real system prompt', async () => {
		const out = await ChatService.normalizeMessagesForApi([
			{ content: 'You are terse.', role: MessageRole.SYSTEM },
			{ content: 'hi', role: MessageRole.USER }
		]);

		expect(out.map((m) => m.content)).toEqual(['You are terse.', 'hi']);
	});
});
