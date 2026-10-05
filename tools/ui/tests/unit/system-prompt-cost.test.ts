import { MessageRole } from '$lib/enums';
import { estimateReprocessCost, formatReprocessCost } from '$lib/utils/system-prompt-cost';
import { describe, expect, it } from 'vitest';

const msg = (role: string, extra: Partial<DatabaseMessage> = {}): DatabaseMessage =>
	({
		children: [],
		content: '',
		convId: 'c',
		id: Math.random().toString(),
		parent: null,
		role,
		timestamp: 0,
		type: 'text',
		...extra
	}) as DatabaseMessage;

describe('estimateReprocessCost', () => {
	it('uses the largest reported context and the largest measured prefill', () => {
		const cost = estimateReprocessCost([
			msg(MessageRole.USER, { content: 'hi' }),
			msg(MessageRole.ASSISTANT, {
				timings: { cache_n: 0, predicted_n: 100, prompt_ms: 60000, prompt_n: 30000 }
			}),
			msg(MessageRole.USER, { content: 'more' }),
			msg(MessageRole.ASSISTANT, {
				timings: { cache_n: 30100, predicted_n: 400, prompt_ms: 500, prompt_n: 200 }
			})
		]);

		expect(cost?.tokens).toBe(30700);
		expect(cost?.seconds).toBeCloseTo(30700 / 500, 5);
		expect(formatReprocessCost(cost!)).toBe('~31k tokens, ~61 s');
	});

	it('falls back to characters and a default speed without timings', () => {
		const cost = estimateReprocessCost([msg(MessageRole.USER, { content: 'x'.repeat(40000) })]);

		expect(cost?.tokens).toBe(10000);
		expect(formatReprocessCost(cost!)).toBe('~10k tokens, ~33 s');
	});

	it('is null without user messages', () => {
		expect(estimateReprocessCost([msg(MessageRole.SYSTEM, { content: 'rules' })])).toBeNull();
	});
});
