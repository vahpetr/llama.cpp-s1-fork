import { CONTEXT_RESERVE } from '$lib/constants/context-reserve.constants';
import {
	driftContinuationBudget,
	driftContinuationSampling,
	driftGuardDecision
} from '$lib/utils/drift-guard';
import { describe, expect, it } from 'vitest';

describe('drift guard: the continuation budget', () => {
	it('remaining = budget - generated (predicted_n; the suppressed </think> is not counted)', () => {
		expect(driftContinuationBudget({ budget: 4000, generated: 1000 })).toEqual({ maxTokens: 3000 });
		expect(driftContinuationBudget({ budget: 1000, generated: 999 })).toEqual({ maxTokens: 1 });
	});

	it('never 0 or negative (0 = unlimited on the wire): skipped instead', () => {
		expect(driftContinuationBudget({ budget: 1000, generated: 1000 })).toEqual({
			skip: 'no output budget left'
		});
		expect(driftContinuationBudget({ budget: 1000, generated: 5000 })).toEqual({
			skip: 'no output budget left'
		});
	});

	it('a known budget with an unknown generated count is skipped', () => {
		expect(driftContinuationBudget({ budget: 1000 })).toEqual({
			skip: 'the generated token count is unknown'
		});
	});

	it('unlimited (none / 0 / -1) = the bounded fallback', () => {
		for (const budget of [undefined, 0, -1]) {
			expect(driftContinuationBudget({ budget, generated: 10 })).toEqual({
				maxTokens: CONTEXT_RESERVE.GUARD_MAX_TOKENS
			});
		}
	});
});

describe('drift guard: sampling and the single retry', () => {
	it('the card Instruct preset (a copy)', () => {
		const s = driftContinuationSampling();

		expect(s).toEqual({
			min_p: 0,
			presence_penalty: 1.5,
			repeat_penalty: 1.0,
			temperature: 0.7,
			top_k: 20,
			top_p: 0.8
		});
		s.temperature = 2;
		expect(driftContinuationSampling().temperature).toBe(0.7);
	});

	it('the Instruct preset as Settings resolve it (the user edits of that preset apply)', () => {
		expect(driftContinuationSampling({ instruct: { temperature: 0.5 } })).toMatchObject({
			temperature: 0.5,
			top_p: 0.8
		});
		// the Thinking preset does not leak in
		expect(driftContinuationSampling({ thinking: { temperature: 1.2 } }).temperature).toBe(0.7);
	});

	it('continue once, then "again"; other stops are not the guard', () => {
		const st = { finish: 'stop', stopType: 'think_reenter' };

		expect(driftGuardDecision(st)).toBe('continue');
		expect(driftGuardDecision({ ...st, state: { note: '', retried: false } })).toBe('continue');
		expect(driftGuardDecision({ ...st, state: { note: '', retried: true } })).toBe('again');
		expect(driftGuardDecision({ finish: 'stop' })).toBe('none');
		expect(driftGuardDecision({ finish: 'length', stopType: 'think_reenter' })).toBe('none');
	});
});
