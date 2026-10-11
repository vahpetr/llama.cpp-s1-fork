import { afterEach, beforeAll, describe, expect, it, vi } from 'vitest';

beforeAll(() => {
	const store = new Map<string, string>();

	(globalThis as unknown as { localStorage: Storage }).localStorage = {
		clear: () => store.clear(),
		getItem: (k: string) => store.get(k) ?? null,
		key: () => null,
		get length() {
			return store.size;
		},
		removeItem: (k: string) => void store.delete(k),
		setItem: (k: string, v: string) => void store.set(k, String(v))
	} as Storage;
});

import {
	effectiveThinking,
	nativeReasoningEffort,
	REASONING_EFFORT_TOKENS,
	SAMPLING_PRESETS,
	samplingPresetLabel
} from '$lib/constants';
import { ReasoningEffort } from '$lib/enums';
import { ChatService } from '$lib/services/chat.service';
import { chatStore } from '$lib/stores/chat/index.svelte';
import { conversationsStore } from '$lib/stores/conversations/index.svelte';
import { modelsStore } from '$lib/stores/models/index.svelte';
import { settingsStore } from '$lib/stores/settings/index.svelte';

afterEach(() => {
	vi.restoreAllMocks();
	vi.unstubAllGlobals();
});

/** the body the real request builder sends for the store's options */
async function bodyFor(
	effort: ReasoningEffort,
	supported: boolean
): Promise<Record<string, unknown>> {
	vi.spyOn(conversationsStore.preferences, 'getReasoningEffort').mockReturnValue(effort);
	vi.spyOn(modelsStore.props, 'supportsReasoningEffort').mockReturnValue(supported);

	const bodies: Record<string, unknown>[] = [];

	vi.stubGlobal(
		'fetch',
		vi.fn(async (_u: string, init?: RequestInit) => {
			bodies.push(JSON.parse(String(init?.body)));

			return new Response('{}', { status: 500 });
		})
	);
	await ChatService.sendMessage([{ content: 'x', role: 'user' }], {
		...chatStore.getApiOptions(),
		stream: false
	} as never).catch(() => {});

	return bodies[0];
}

describe('native reasoning_effort (template caps)', () => {
	it('maps Low / Medium / High / Max to low / medium / xhigh / xhigh (the template takes only these); Default / Off send none', () => {
		expect(nativeReasoningEffort(ReasoningEffort.LOW, true)).toBe('low');
		expect(nativeReasoningEffort(ReasoningEffort.MEDIUM, true)).toBe('medium');
		expect(nativeReasoningEffort(ReasoningEffort.HIGH, true)).toBe('xhigh');
		expect(nativeReasoningEffort(ReasoningEffort.MAX, true)).toBe('xhigh');
		expect(nativeReasoningEffort(ReasoningEffort.DEFAULT, true)).toBeUndefined();
		expect(nativeReasoningEffort(ReasoningEffort.OFF, true)).toBeUndefined();
		expect(nativeReasoningEffort(ReasoningEffort.HIGH, false)).toBeUndefined();
	});

	it('sends reasoning_effort with the unchanged kwarg and budget when the caps say so', async () => {
		for (const [effort, native] of [
			[ReasoningEffort.LOW, 'low'],
			[ReasoningEffort.MEDIUM, 'medium'],
			[ReasoningEffort.HIGH, 'xhigh'],
			[ReasoningEffort.MAX, 'xhigh']
		] as const) {
			const body = await bodyFor(effort, true);

			expect(body.reasoning_effort).toBe(native);
			expect(body.chat_template_kwargs).toEqual({ enable_thinking: true });
			expect(body.thinking_budget_tokens).toBe(
				REASONING_EFFORT_TOKENS[effort] >= 0 ? REASONING_EFFORT_TOKENS[effort] : undefined
			);
			vi.restoreAllMocks();
		}
	});

	it('no reasoning_effort when the template does not take it (budget mapping unchanged)', async () => {
		const body = await bodyFor(ReasoningEffort.HIGH, false);

		expect(body.reasoning_effort).toBeUndefined();
		expect(body).toMatchObject({
			chat_template_kwargs: { enable_thinking: true },
			thinking_budget_tokens: 8192
		});
	});

	it('Off keeps enable_thinking=false only; Default sends nothing', async () => {
		const off = await bodyFor(ReasoningEffort.OFF, true);

		expect(off.chat_template_kwargs).toEqual({ enable_thinking: false });
		expect(off.reasoning_effort).toBeUndefined();
		expect(off.thinking_budget_tokens).toBeUndefined();
		vi.restoreAllMocks();

		const def = await bodyFor(ReasoningEffort.DEFAULT, true);

		expect(def.chat_template_kwargs).toBeUndefined();
		expect(def.reasoning_effort).toBeUndefined();
		expect(def.thinking_budget_tokens).toBeUndefined();
	});
});

describe('sampling presets by thinking state (model card best practices)', () => {
	const unset = {
		min_p: undefined,
		presence_penalty: undefined,
		repeat_penalty: undefined,
		temperature: undefined,
		top_k: undefined,
		top_p: undefined
	};

	it('thinking on (any level) -> 1.0 / 0.95 / 20 / 0 / 0.0 / 1.0', async () => {
		settingsStore.config = { ...settingsStore.config, ...unset };

		const body = await bodyFor(ReasoningEffort.LOW, false);

		expect(body).toMatchObject({
			min_p: 0,
			presence_penalty: 0,
			repeat_penalty: 1,
			temperature: 1,
			top_k: 20,
			top_p: 0.95
		});
	});

	it('Off, and Default under --reasoning off -> the instruct preset 0.7 / 0.8 / 20 / 0 / 1.5 / 1.0', async () => {
		settingsStore.config = { ...settingsStore.config, ...unset };

		for (const effort of [ReasoningEffort.OFF, ReasoningEffort.DEFAULT]) {
			const body = await bodyFor(effort, true);

			expect(body).toMatchObject({
				min_p: 0,
				presence_penalty: 1.5,
				repeat_penalty: 1,
				temperature: 0.7,
				top_k: 20,
				top_p: 0.8
			});
			vi.restoreAllMocks();
		}
	});

	it('explicit user values win over the preset', async () => {
		settingsStore.config = {
			...settingsStore.config,
			...unset,
			presence_penalty: 0.3,
			temperature: 0.2
		};

		const body = await bodyFor(ReasoningEffort.HIGH, true);

		expect(body).toMatchObject({ presence_penalty: 0.3, temperature: 0.2, top_k: 20, top_p: 0.95 });
		settingsStore.config = { ...settingsStore.config, ...unset };
	});

	it('Default follows the server default thinking; the lightbulb title names the preset', () => {
		expect(effectiveThinking(ReasoningEffort.DEFAULT, false)).toBe(false);
		expect(effectiveThinking(ReasoningEffort.DEFAULT, true)).toBe(true);
		expect(effectiveThinking(ReasoningEffort.OFF, true)).toBe(false);
		expect(effectiveThinking(ReasoningEffort.LOW, false)).toBe(true);
		expect(
			samplingPresetLabel({
				native: 'xhigh',
				temperature: SAMPLING_PRESETS.thinking.temperature,
				thinking: true
			})
		).toBe('thinking · xhigh · T 1.0');
		expect(samplingPresetLabel({ temperature: 0.7, thinking: false })).toBe('instruct · T 0.7');
	});
});

describe('reasoning_effort fallback', () => {
	it('a 400 naming reasoning effort: retried once without it, never sent to that model again', async () => {
		vi.spyOn(conversationsStore.preferences, 'getReasoningEffort').mockReturnValue(
			ReasoningEffort.MEDIUM
		);
		modelsStore.props.reasoningEffortRejected.clear();

		const caps = vi.spyOn(modelsStore.props, 'supportsReasoningEffort');
		const bodies: Record<string, unknown>[] = [];

		vi.stubGlobal(
			'fetch',
			vi.fn(async (_u: string, init?: RequestInit) => {
				const body = JSON.parse(String(init?.body));

				bodies.push(body);

				return body.reasoning_effort
					? new Response(
							JSON.stringify({
								error: {
									code: 400,
									message:
										'Unexpected reasoning effort medium. Supported types are xhigh (default), medium, and low.'
								}
							}),
							{ status: 400 }
						)
					: new Response(JSON.stringify({ choices: [{ message: { content: 'ok' } }] }));
			})
		);

		caps.mockReturnValue(true);
		await ChatService.sendMessage([{ content: 'x', role: 'user' }], {
			...chatStore.getApiOptions(),
			stream: false
		} as never);

		expect(bodies).toHaveLength(2);
		expect(bodies[0].reasoning_effort).toBe('medium');
		expect(bodies[1].reasoning_effort).toBeUndefined();
		expect(bodies[1].chat_template_kwargs).toEqual({ enable_thinking: true });

		// remembered: the real check now says no
		caps.mockRestore();
		expect(modelsStore.props.reasoningEffortRejected.size).toBe(1);
		expect(modelsStore.props.supportsReasoningEffort()).toBe(false);
		modelsStore.props.reasoningEffortRejected.clear();
	});
});
