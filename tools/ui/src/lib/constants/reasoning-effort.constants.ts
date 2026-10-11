import { ReasoningEffort } from '$lib/enums';
import type { ReasoningEffortLevel } from '$lib/types';

/**
 * Reasoning effort UI labels.
 * Keys match the ReasoningEffort enum values for type-safe lookups.
 */
export const REASONING_EFFORT_LABELS: Record<string, string> = {
	[ReasoningEffort.DEFAULT]: 'Default',
	[ReasoningEffort.HIGH]: 'High',
	[ReasoningEffort.LOW]: 'Low',
	[ReasoningEffort.MAX]: 'Max',
	[ReasoningEffort.MEDIUM]: 'Medium',
	[ReasoningEffort.OFF]: 'Off'
};

export const REASONING_EFFORT_LEVELS: ReasoningEffortLevel[] = [
	{ label: 'Default', value: ReasoningEffort.DEFAULT },
	{ label: 'Off', value: ReasoningEffort.OFF },
	{ label: 'Low', value: ReasoningEffort.LOW },
	{ label: 'Medium', value: ReasoningEffort.MEDIUM },
	{ label: 'High', value: ReasoningEffort.HIGH },
	{ hasInfo: true, label: 'Max', value: ReasoningEffort.MAX }
];

/**
 * Reasoning effort to the native OAI `reasoning_effort` value, sent only when the chat template
 * supports it (`/props` chat_template_caps.supports_reasoning_effort); Default / Off send none.
 * Qwen3.8 takes only xhigh (its default), medium and low and raises on anything else, so High and
 * Max both map to xhigh - they still differ by the thinking budget (8192 vs unlimited).
 */
export const REASONING_EFFORT_OAI: Record<string, string> = {
	[ReasoningEffort.HIGH]: 'xhigh',
	[ReasoningEffort.LOW]: 'low',
	[ReasoningEffort.MAX]: 'xhigh',
	[ReasoningEffort.MEDIUM]: 'medium'
};

/**
 * Sampling presets by thinking state (the Qwen3.8 model card "Best Practices"), sent per request
 * only for the parameters the user left unset in Settings - explicit values win.
 */
export const SAMPLING_PRESETS = {
	instruct: {
		min_p: 0,
		presence_penalty: 1.5,
		repeat_penalty: 1.0,
		temperature: 0.7,
		top_k: 20,
		top_p: 0.8
	},
	thinking: {
		min_p: 0,
		presence_penalty: 0.0,
		repeat_penalty: 1.0,
		temperature: 1.0,
		top_k: 20,
		top_p: 0.95
	}
} as const;

/** The six sampling fields a preset sets. */
export const PRESET_FIELDS = [
	'temperature',
	'top_p',
	'top_k',
	'min_p',
	'presence_penalty',
	'repeat_penalty'
] as const;

export type PresetField = (typeof PRESET_FIELDS)[number];
export type SamplingPresetValues = Record<PresetField, number>;
export type SamplingPresetOverrides = Partial<Record<PresetField, number>>;

/**
 * The user's edits of the presets (Settings -> Sampling & Penalties): the Thinking and Instruct
 * presets (a missing preset = the model card) and optional per-level overrides of Thinking.
 */
export interface SamplingPresetSettings {
	thinking?: SamplingPresetOverrides;
	instruct?: SamplingPresetOverrides;
	levels?: Partial<Record<'low' | 'medium' | 'high' | 'max', SamplingPresetOverrides>>;
}

export type SamplingPresetSource = 'card' | 'custom' | 'level';

/** A valid value of a preset field: a finite number, an integer for top_k; else undefined. */
export function sanitizePresetValue(field: PresetField, v: unknown): number | undefined {
	if (typeof v !== 'number' || !Number.isFinite(v)) return undefined;

	if (field === 'top_k' && !Number.isInteger(v)) return undefined;

	return v;
}

/** Only valid values of the six fields. */
export function cleanPresetOverrides(o: unknown): SamplingPresetOverrides {
	const out: SamplingPresetOverrides = {};

	if (!o || typeof o !== 'object') return out;

	for (const k of PRESET_FIELDS) {
		const v = sanitizePresetValue(k, (o as Record<string, unknown>)[k]);

		if (v !== undefined) out[k] = v;
	}

	return out;
}

const PRESET_LEVELS = ['low', 'medium', 'high', 'max'] as const;

/** Stored / imported preset settings with every invalid part dropped. */
export function cleanPresetSettings(o: unknown): SamplingPresetSettings {
	const out: SamplingPresetSettings = {};

	if (!o || typeof o !== 'object' || Array.isArray(o)) return out;

	const src = o as Record<string, unknown>;

	for (const state of ['thinking', 'instruct'] as const) {
		const c = cleanPresetOverrides(src[state]);

		if (Object.keys(c).length > 0) out[state] = c;
	}

	const levels: NonNullable<SamplingPresetSettings['levels']> = {};

	for (const level of PRESET_LEVELS) {
		const c = cleanPresetOverrides((src.levels as Record<string, unknown> | undefined)?.[level]);

		if (Object.keys(c).length > 0) levels[level] = c;
	}

	if (Object.keys(levels).length > 0) out.levels = levels;

	return out;
}

/**
 * The presets a settings file brings: cleaned, or null when the file has none (an older export) -
 * then the presets here stay as they are.
 */
export function importedPresets(
	data: { samplingPresets?: unknown } | null | undefined
): SamplingPresetSettings | null {
	const p = data?.samplingPresets;

	return p && typeof p === 'object' && !Array.isArray(p) ? cleanPresetSettings(p) : null;
}

/**
 * A copy of preset settings with one field of a preset (`thinking` / `instruct`) or of a level's
 * overrides set - or cleared (undefined / invalid: the card value / the Thinking preset applies).
 */
export function withPresetField(
	settings: SamplingPresetSettings,
	target: 'thinking' | 'instruct' | 'low' | 'medium' | 'high' | 'max',
	field: PresetField,
	value: number | undefined
): SamplingPresetSettings {
	const isLevel = target !== 'thinking' && target !== 'instruct';
	const current: SamplingPresetOverrides = {
		...(isLevel
			? settings.levels?.[target as (typeof PRESET_LEVELS)[number]]
			: settings[target as 'thinking' | 'instruct'])
	};
	const v = sanitizePresetValue(field, value);

	if (v === undefined) delete current[field];
	else current[field] = v;

	return isLevel
		? { ...settings, levels: { ...settings.levels, [target]: current } }
		: { ...settings, [target]: current };
}

/** A copy of preset settings with a preset back to the model card (the level overrides stay). */
export function withoutPreset(
	settings: SamplingPresetSettings,
	state: 'thinking' | 'instruct'
): SamplingPresetSettings {
	const next = { ...settings };

	delete next[state];

	return next;
}

/** A preset's values: the model card with the user's edits (fields that differ from the card). */
export function presetValues(
	state: 'thinking' | 'instruct',
	settings: SamplingPresetSettings | undefined
): { values: SamplingPresetValues; custom: boolean } {
	const card = SAMPLING_PRESETS[state];
	const edits = cleanPresetOverrides(settings?.[state]);
	const values = { ...card, ...edits } as SamplingPresetValues;

	return { custom: PRESET_FIELDS.some((k) => values[k] !== card[k]), values };
}

/**
 * The sampling a request gets from the presets (before the user's own Sampling values, which win):
 * none for Default or a model without thinking; Off -> the Instruct preset; a level -> the
 * Thinking preset with that level's overrides on top. `source` = card / custom / level.
 */
export function resolveSamplingPreset(
	effort: string,
	modelSupportsThinking: boolean,
	settings?: SamplingPresetSettings
): {
	values: SamplingPresetValues;
	source: SamplingPresetSource;
	state: 'thinking' | 'instruct';
} | null {
	if (!modelSupportsThinking || effort === ReasoningEffort.DEFAULT) return null;

	if (effort === ReasoningEffort.OFF) {
		const p = presetValues('instruct', settings);

		return { source: p.custom ? 'custom' : 'card', state: 'instruct', values: p.values };
	}

	const p = presetValues('thinking', settings);
	const level = cleanPresetOverrides(
		settings?.levels?.[effort as 'low' | 'medium' | 'high' | 'max']
	);
	const hasLevel = Object.keys(level).length > 0;

	return {
		source: hasLevel ? 'level' : p.custom ? 'custom' : 'card',
		state: 'thinking',
		values: { ...p.values, ...level }
	};
}

/** The lightbulb title: "thinking · xhigh · T 1.0 · preset: card" / "instruct · T 0.7". */
export function samplingPresetLabel(p: {
	thinking: boolean;
	native?: string;
	temperature?: number;
	source?: SamplingPresetSource;
}): string {
	return [
		p.thinking ? 'thinking' : 'instruct',
		p.native,
		p.temperature !== undefined ? `T ${p.temperature.toFixed(1)}` : '',
		p.source === 'level' ? 'level override' : p.source ? `preset: ${p.source}` : ''
	]
		.filter(Boolean)
		.join(' · ');
}

/**
 * Is a 400 error the template refusing the native effort? Only its own message ("Unexpected
 * reasoning effort ... Supported types are xhigh (default), medium, and low."), nothing broader.
 */
export function isReasoningEffortRejection(message: string): boolean {
	if (/unexpected reasoning effort/i.test(message)) return true;

	// the "supported ... xhigh" form counts only when the message is about the reasoning effort
	return (
		/\b(reasoning|effort)\b/i.test(message) && /supported (types|levels)[^.]*xhigh/i.test(message)
	);
}

/** The native `reasoning_effort` for a level, or undefined (Default, Off, or not supported). */
export function nativeReasoningEffort(effort: string, supported: boolean): string | undefined {
	return supported ? REASONING_EFFORT_OAI[effort] : undefined;
}

/**
 * Reasoning effort to token budget mapping.
 * Maps the ReasoningEffort enum values to concrete token counts for the server.
 */
export const REASONING_EFFORT_TOKENS: Record<string, number> = {
	[ReasoningEffort.HIGH]: 8192,
	[ReasoningEffort.LOW]: 512,
	[ReasoningEffort.MAX]: -1, // unlimited
	[ReasoningEffort.MEDIUM]: 2048
};

/** Settings: which preset the current chat uses ("Active in this chat: ..."). */
export function activePresetLine(
	effort: string,
	modelSupportsThinking: boolean,
	settings?: SamplingPresetSettings
): string {
	if (!modelSupportsThinking) return 'none — the model does not support thinking';

	const p = resolveSamplingPreset(effort, modelSupportsThinking, settings);

	if (!p) return 'none — Default inherits the server';

	if (p.state === 'instruct') return 'Instruct';

	const level = REASONING_EFFORT_LABELS[effort] ?? effort;

	return `Thinking (level ${level}${p.source === 'level' ? ', with its overrides' : ''})`;
}
