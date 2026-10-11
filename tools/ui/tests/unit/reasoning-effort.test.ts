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
	activePresetLine,
	cleanPresetSettings,
	importedPresets,
	isReasoningEffortRejection,
	nativeReasoningEffort,
	REASONING_EFFORT_TOKENS,
	resolveSamplingPreset,
	SAMPLING_PRESETS,
	samplingPresetLabel,
	sanitizePresetValue,
	withoutPreset,
	withPresetField
} from '$lib/constants';
import { ReasoningEffort } from '$lib/enums';
import { ChatService } from '$lib/services/chat.service';
import { chatStore } from '$lib/stores/chat/index.svelte';
import { conversationsStore } from '$lib/stores/conversations/index.svelte';
import { modelsStore } from '$lib/stores/models/index.svelte';
import { serverStore } from '$lib/stores/server.svelte';
import { settingsStore } from '$lib/stores/settings/index.svelte';
import { applyRequestSnapshot, snapshotRequestOptions } from '$lib/utils/request-snapshot';

const UNSET = {
	min_p: undefined,
	presence_penalty: undefined,
	repeat_penalty: undefined,
	temperature: undefined,
	top_k: undefined,
	top_p: undefined
};

afterEach(() => {
	vi.restoreAllMocks();
	vi.unstubAllGlobals();
	settingsStore.config = { ...settingsStore.config, ...UNSET, customJson: '' };
	settingsStore.userSampling = {};
	settingsStore.samplingPresets = {};
	modelsStore.props.reasoningEffortRejected.clear();
});

/** the body the real request builder sends for the store's options */
async function bodyFor(
	effort: ReasoningEffort,
	supported: boolean,
	thinkingModel = true
): Promise<Record<string, unknown>> {
	vi.spyOn(conversationsStore.preferences, 'getReasoningEffort').mockReturnValue(effort);
	vi.spyOn(modelsStore.props, 'supportsReasoningEffort').mockReturnValue(supported);
	vi.spyOn(modelsStore.props, 'modelSupportsThinking').mockReturnValue(thinkingModel);

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
	it('maps Low / Medium / High / Max to low / medium / xhigh / xhigh; Default / Off send none', () => {
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
		vi.restoreAllMocks();

		const def = await bodyFor(ReasoningEffort.DEFAULT, true);

		expect(def.chat_template_kwargs).toBeUndefined();
		expect(def.reasoning_effort).toBeUndefined();
		expect(def.thinking_budget_tokens).toBeUndefined();
	});
});

describe('sampling presets (astra: Default inherits the server, durable user values)', () => {
	const SAMPLING = ['temperature', 'top_p', 'top_k', 'min_p', 'presence_penalty', 'repeat_penalty'];

	it('an explicit level on a thinking model -> the thinking preset', async () => {
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

	it('explicit Off on a thinking model -> the instruct preset', async () => {
		const body = await bodyFor(ReasoningEffort.OFF, true);

		expect(body).toMatchObject({
			min_p: 0,
			presence_penalty: 1.5,
			repeat_penalty: 1,
			temperature: 0.7,
			top_k: 20,
			top_p: 0.8
		});
	});

	it('Default sends no preset sampling (the server defaults apply)', async () => {
		const body = await bodyFor(ReasoningEffort.DEFAULT, true);

		for (const k of SAMPLING) expect(body[k]).toBeUndefined();
	});

	it('a model that does not support thinking gets no preset at any level', async () => {
		const body = await bodyFor(ReasoningEffort.HIGH, false, false);

		for (const k of SAMPLING) expect(body[k]).toBeUndefined();
	});

	it('explicit user values win', async () => {
		settingsStore.config = { ...settingsStore.config, presence_penalty: 0.3, temperature: 0.2 };

		const body = await bodyFor(ReasoningEffort.HIGH, true);

		expect(body).toMatchObject({ presence_penalty: 0.3, temperature: 0.2, top_p: 0.95 });
	});

	it("a user value equal to the server default stays the user's after the sync clears it", async () => {
		// the user sets presence_penalty 0 (= the server default); the sync clears it from the config
		settingsStore.updateMultipleConfig({ presence_penalty: 0 } as never);
		settingsStore.config = { ...settingsStore.config, presence_penalty: undefined };
		// a later settings save of the untouched (empty) field keeps the provenance
		settingsStore.updateMultipleConfig({ presence_penalty: '' } as never);

		const body = await bodyFor(ReasoningEffort.OFF, true);

		expect(settingsStore.userSampling).toEqual({ presence_penalty: 0 });
		expect(body.presence_penalty).toBe(0);
		expect(body.temperature).toBe(0.7);
	});

	it('clearing a value the user had set forgets it', () => {
		settingsStore.updateMultipleConfig({ temperature: 0.4 } as never);
		expect(settingsStore.userSampling).toEqual({ temperature: 0.4 });
		settingsStore.updateMultipleConfig({ temperature: '' } as never);
		expect(settingsStore.userSampling).toEqual({});
	});

	it('preset rule and the lightbulb label', () => {
		expect(resolveSamplingPreset(ReasoningEffort.DEFAULT, true)).toBeNull();
		expect(resolveSamplingPreset(ReasoningEffort.HIGH, false)).toBeNull();
		expect(resolveSamplingPreset(ReasoningEffort.OFF, true)?.values).toEqual(
			SAMPLING_PRESETS.instruct
		);
		expect(resolveSamplingPreset(ReasoningEffort.MAX, true)?.values).toEqual(
			SAMPLING_PRESETS.thinking
		);
		expect(samplingPresetLabel({ native: 'xhigh', temperature: 1, thinking: true })).toBe(
			'thinking · xhigh · T 1.0'
		);
		expect(samplingPresetLabel({ temperature: 0.7, thinking: false })).toBe('instruct · T 0.7');
	});
});

describe('per-turn template-option snapshot for continuations', () => {
	it("a continuation reuses the turn's options verbatim, whatever the picker says now", () => {
		const snap = snapshotRequestOptions({
			enableThinking: true,
			reasoningEffort: 'medium',
			reasoningEffortNative: 'medium',
			stream: true,
			temperature: 1,
			top_p: 0.95
		});

		expect(snap).toEqual({
			enableThinking: true,
			reasoningEffort: 'medium',
			reasoningEffortNative: 'medium',
			sampling: { temperature: 1, top_p: 0.95 }
		});

		// now: Off with the instruct preset
		const now = {
			enableThinking: false,
			presence_penalty: 1.5,
			stream: true,
			temperature: 0.7,
			top_p: 0.8
		};
		const out = applyRequestSnapshot(now, snap);

		expect(out).toEqual({
			enableThinking: true,
			reasoningEffort: 'medium',
			reasoningEffortNative: 'medium',
			stream: true,
			temperature: 1,
			top_p: 0.95
		});
		// a snapshot without a native effort removes the current one
		expect(applyRequestSnapshot({ reasoningEffortNative: 'low', x: 1 }, { sampling: {} })).toEqual({
			x: 1
		});
		expect(applyRequestSnapshot(now, undefined)).toBe(now);
	});

	it('the continuation request body carries the snapshot, not the current level', async () => {
		vi.spyOn(conversationsStore.preferences, 'getReasoningEffort').mockReturnValue(
			ReasoningEffort.OFF
		);
		vi.spyOn(modelsStore.props, 'modelSupportsThinking').mockReturnValue(true);
		vi.spyOn(modelsStore.props, 'supportsReasoningEffort').mockReturnValue(true);

		const bodies: Record<string, unknown>[] = [];

		vi.stubGlobal(
			'fetch',
			vi.fn(async (_u: string, init?: RequestInit) => {
				bodies.push(JSON.parse(String(init?.body)));

				return new Response('{}', { status: 500 });
			})
		);
		await ChatService.sendMessage(
			[
				{ content: 'q', role: 'user' },
				{ content: 'partial', role: 'assistant' }
			],
			{
				...applyRequestSnapshot(chatStore.getApiOptions(), {
					enableThinking: true,
					reasoningEffort: ReasoningEffort.LOW,
					reasoningEffortNative: 'low',
					sampling: { temperature: 1, top_p: 0.95 }
				}),
				continueFinalMessage: true,
				stream: false
			} as never
		).catch(() => {});

		expect(bodies[0]).toMatchObject({
			chat_template_kwargs: { enable_thinking: true },
			continue_final_message: true,
			reasoning_effort: 'low',
			temperature: 1,
			top_p: 0.95
		});
	});
});

describe('reasoning_effort fallback (narrow, per model)', () => {
	it("only the template's own refusal counts", () => {
		expect(
			isReasoningEffortRejection(
				'Unexpected reasoning effort high. Supported types are xhigh (default), medium, and low.'
			)
		).toBe(true);
		expect(
			isReasoningEffortRejection('Reasoning effort: supported levels are xhigh, medium, low')
		).toBe(true);
		// the "supported ... xhigh" form without a reasoning / effort reference does not count
		expect(isReasoningEffortRejection('Supported levels: xhigh, medium, low')).toBe(false);
		expect(
			isReasoningEffortRejection('Supported types are xhigh (default), medium, and low.')
		).toBe(false);
		expect(isReasoningEffortRejection('invalid reasoning_effort type')).toBe(false);
		expect(isReasoningEffortRejection('the prompt is too long for the reasoning effort')).toBe(
			false
		);
	});

	it('one retry without the field (also when the custom JSON re-adds it), remembered per model, cleared by a /props refresh', async () => {
		vi.spyOn(conversationsStore.preferences, 'getReasoningEffort').mockReturnValue(
			ReasoningEffort.MEDIUM
		);
		vi.spyOn(modelsStore.props, 'modelSupportsThinking').mockReturnValue(true);
		vi.spyOn(serverStore, 'isRouterMode', 'get').mockReturnValue(false);

		const propsA = {
			chat_template_caps: { supports_reasoning_effort: true },
			model_alias: 'brain'
		};

		serverStore.props = propsA as never;
		settingsStore.config = {
			...settingsStore.config,
			customJson: '{"reasoning_effort":"medium"}'
		};

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

		await ChatService.sendMessage([{ content: 'x', role: 'user' }], {
			...chatStore.getApiOptions(),
			stream: false
		} as never);

		expect(bodies).toHaveLength(2);
		expect(bodies[0].reasoning_effort).toBe('medium');
		expect(bodies[1].reasoning_effort).toBeUndefined();
		expect(bodies[1].chat_template_kwargs).toEqual({ enable_thinking: true });
		expect([...modelsStore.props.reasoningEffortRejected.keys()]).toEqual(['brain']);
		expect(modelsStore.props.supportsReasoningEffort()).toBe(false);

		// a refreshed /props (another object) clears the memory
		serverStore.props = { ...propsA } as never;
		expect(modelsStore.props.supportsReasoningEffort()).toBe(true);
		serverStore.props = null;
	});

	it("llama-server's HTTP 500 template refusal (Jinja exception) also retries once without it; other 500s do not", async () => {
		vi.spyOn(conversationsStore.preferences, 'getReasoningEffort').mockReturnValue(
			ReasoningEffort.LOW
		);
		vi.spyOn(modelsStore.props, 'modelSupportsThinking').mockReturnValue(true);
		vi.spyOn(modelsStore.props, 'supportsReasoningEffort').mockReturnValue(true);

		const bodies: Record<string, unknown>[] = [];

		let message =
			'Jinja Exception: Unexpected reasoning effort low. Supported types are xhigh (default), medium, and low.';

		vi.stubGlobal(
			'fetch',
			vi.fn(async (_u: string, init?: RequestInit) => {
				const body = JSON.parse(String(init?.body));

				bodies.push(body);

				return body.reasoning_effort
					? new Response(JSON.stringify({ error: { code: 500, message } }), { status: 500 })
					: new Response(JSON.stringify({ choices: [{ message: { content: 'ok' } }] }));
			})
		);

		await ChatService.sendMessage([{ content: 'x', role: 'user' }], {
			...chatStore.getApiOptions(),
			stream: false
		} as never);

		expect(bodies.map((b) => b.reasoning_effort)).toEqual(['low', undefined]);
		expect(modelsStore.props.reasoningEffortRejected.size).toBe(1);

		// a 500 that is not the template's refusal: no retry
		modelsStore.props.reasoningEffortRejected.clear();
		bodies.length = 0;
		message = 'Internal server error: slot unavailable';
		await ChatService.sendMessage([{ content: 'x', role: 'user' }], {
			...chatStore.getApiOptions(),
			stream: false
		} as never).catch(() => {});

		expect(bodies.map((b) => b.reasoning_effort)).toEqual(['low']);
		expect(modelsStore.props.reasoningEffortRejected.size).toBe(0);
	});
});

describe('editable presets by reasoning state + per-level overrides', () => {
	it('precedence: user value > level override > edited preset; Default sends nothing', async () => {
		settingsStore.setPresetField('thinking', 'temperature', 0.9);
		settingsStore.setPresetField('thinking', 'top_p', 0.9);
		settingsStore.setPresetField('medium', 'top_p', 0.85);
		settingsStore.setPresetField('medium', 'min_p', 0.05);
		settingsStore.config = { ...settingsStore.config, min_p: 0.02 };

		const medium = await bodyFor(ReasoningEffort.MEDIUM, false);

		expect(medium).toMatchObject({
			min_p: 0.02, // the user's own Sampling value wins over the level override
			presence_penalty: 0,
			temperature: 0.9, // the edited Thinking preset
			top_k: 20,
			top_p: 0.85 // the Medium override
		});
		vi.restoreAllMocks();

		const high = await bodyFor(ReasoningEffort.HIGH, false);

		expect(high).toMatchObject({ temperature: 0.9, top_p: 0.9 }); // no High override
		vi.restoreAllMocks();

		const def = await bodyFor(ReasoningEffort.DEFAULT, false);

		expect(def.temperature).toBeUndefined();
		expect(def.top_p).toBeUndefined();
	});

	it('the source: card, custom, level; reset to the model card', () => {
		expect(resolveSamplingPreset(ReasoningEffort.LOW, true, {})?.source).toBe('card');
		settingsStore.setPresetField('thinking', 'temperature', 0.8);
		expect(
			resolveSamplingPreset(ReasoningEffort.LOW, true, settingsStore.samplingPresets)?.source
		).toBe('custom');
		settingsStore.setPresetField('low', 'top_k', 40);
		expect(
			resolveSamplingPreset(ReasoningEffort.LOW, true, settingsStore.samplingPresets)?.source
		).toBe('level');
		// an edit equal to the card is not "custom"
		settingsStore.setPresetField('instruct', 'temperature', 0.7);
		expect(
			resolveSamplingPreset(ReasoningEffort.OFF, true, settingsStore.samplingPresets)?.source
		).toBe('card');

		settingsStore.resetPreset('thinking');
		expect(settingsStore.samplingPresets.thinking).toBeUndefined();
		expect(
			resolveSamplingPreset(ReasoningEffort.MEDIUM, true, settingsStore.samplingPresets)
		).toEqual({
			source: 'card',
			state: 'thinking',
			values: SAMPLING_PRESETS.thinking
		});
		// clearing a level field inherits again
		settingsStore.setPresetField('low', 'top_k', undefined);
		expect(
			resolveSamplingPreset(ReasoningEffort.LOW, true, settingsStore.samplingPresets)?.source
		).toBe('card');
	});

	it('the lightbulb title names the source; the settings line names the active preset', () => {
		expect(
			samplingPresetLabel({ native: 'medium', source: 'card', temperature: 1, thinking: true })
		).toBe('thinking · medium · T 1.0 · preset: card');
		expect(samplingPresetLabel({ source: 'custom', temperature: 0.6, thinking: false })).toBe(
			'instruct · T 0.6 · preset: custom'
		);
		expect(samplingPresetLabel({ source: 'level', temperature: 1, thinking: true })).toBe(
			'thinking · T 1.0 · level override'
		);

		expect(activePresetLine(ReasoningEffort.MEDIUM, true, {})).toBe('Thinking (level Medium)');
		expect(activePresetLine(ReasoningEffort.OFF, true, {})).toBe('Instruct');
		expect(activePresetLine(ReasoningEffort.DEFAULT, true, {})).toBe(
			'none — Default inherits the server'
		);
		expect(
			activePresetLine(ReasoningEffort.HIGH, true, { levels: { high: { temperature: 0.5 } } })
		).toBe('Thinking (level High, with its overrides)');
	});

	it('stored presets are cleaned: junk and missing fields fall back to the card', () => {
		expect(
			resolveSamplingPreset(ReasoningEffort.OFF, true, {
				instruct: { temperature: 'hot' as never, top_p: Number.NaN }
			})?.values
		).toEqual(SAMPLING_PRESETS.instruct);
	});
});

describe('astra 0223 review: draft, export/import, reset, integer top_k, hasValue', () => {
	it('top_k must be an integer (sanitizer, draft edits, stored / imported settings)', () => {
		expect(sanitizePresetValue('top_k', 20.5)).toBeUndefined();
		expect(sanitizePresetValue('top_k', 40)).toBe(40);
		expect(sanitizePresetValue('temperature', 0.55)).toBe(0.55);
		expect(withPresetField({}, 'thinking', 'top_k', 20.5)).toEqual({ thinking: {} });
		expect(
			cleanPresetSettings({ levels: { low: { top_k: 3.2, top_p: 0.9 } }, thinking: { top_k: 'x' } })
		).toEqual({ levels: { low: { top_p: 0.9 } } });
	});

	it('draft helpers never touch the stored presets (Save persists, Cancel drops)', () => {
		settingsStore.setSamplingPresets({ thinking: { temperature: 0.9 } });

		const draft = withPresetField(
			withoutPreset(settingsStore.samplingPresets, 'thinking'),
			'medium',
			'top_p',
			0.8
		);

		expect(draft).toEqual({ levels: { medium: { top_p: 0.8 } } });
		// nothing persisted until Save
		expect(settingsStore.samplingPresets).toEqual({ thinking: { temperature: 0.9 } });
		settingsStore.setSamplingPresets(draft);
		expect(settingsStore.samplingPresets).toEqual({ levels: { medium: { top_p: 0.8 } } });
	});

	it('export carries the presets; import replaces them only when the file has them', () => {
		settingsStore.setSamplingPresets({ instruct: { temperature: 0.6 } });

		const exported = settingsStore.exportSettings();

		expect(exported.samplingPresets).toEqual({ instruct: { temperature: 0.6 } });
		// an older file without presets: none imported (the presets here stay)
		expect(importedPresets({})).toBeNull();
		expect(importedPresets({ samplingPresets: [] })).toBeNull();
		// a file with presets: cleaned (a fractional top_k dropped)
		expect(
			importedPresets({ samplingPresets: { instruct: { temperature: 0.65, top_k: 1.5 } } })
		).toEqual({ instruct: { temperature: 0.65 } });
	});

	it('"Reset to default" (forceSyncWithServerDefaults) and resetConfig clear the presets', () => {
		settingsStore.setSamplingPresets({
			levels: { max: { temperature: 1.2 } },
			thinking: { top_p: 0.9 }
		});
		settingsStore.forceSyncWithServerDefaults();
		expect(settingsStore.samplingPresets).toEqual({});
		settingsStore.setSamplingPresets({ instruct: { top_p: 0.7 } });
		settingsStore.resetConfig();
		expect(settingsStore.samplingPresets).toEqual({});
	});

	it('hasValue ignores whitespace-only strings and NaN (no NaN / blank sampling sent)', async () => {
		settingsStore.config = { ...settingsStore.config, temperature: '   ', top_p: Number.NaN };

		const body = await bodyFor(ReasoningEffort.OFF, true);

		// both fall back to the instruct preset instead of NaN / blank
		expect(body.temperature).toBe(0.7);
		expect(body.top_p).toBe(0.8);
	});

	it('a fractional top_k from an old setting is sent as an integer', async () => {
		settingsStore.config = { ...settingsStore.config, top_k: 20.6 };

		const body = await bodyFor(ReasoningEffort.DEFAULT, true);

		expect(body.top_k).toBe(21);
	});
});
