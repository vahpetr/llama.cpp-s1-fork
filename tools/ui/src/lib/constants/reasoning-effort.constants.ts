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

/**
 * The sampling preset a request gets: only for an explicit choice on a model that supports
 * thinking - a level -> thinking, Off -> instruct; Default (and non-thinking models) -> none, the
 * server's own sampling applies (/props carries no thinking-activation signal).
 */
export function samplingPresetFor(
	effort: string,
	modelSupportsThinking: boolean
): (typeof SAMPLING_PRESETS)['thinking'] | (typeof SAMPLING_PRESETS)['instruct'] | null {
	if (!modelSupportsThinking || effort === ReasoningEffort.DEFAULT) return null;

	return effort === ReasoningEffort.OFF ? SAMPLING_PRESETS.instruct : SAMPLING_PRESETS.thinking;
}

/** The lightbulb title: "thinking · xhigh · T 1.0" / "instruct · T 0.7". */
export function samplingPresetLabel(p: {
	thinking: boolean;
	native?: string;
	temperature?: number;
}): string {
	return [
		p.thinking ? 'thinking' : 'instruct',
		p.native,
		p.temperature !== undefined ? `T ${p.temperature.toFixed(1)}` : ''
	]
		.filter(Boolean)
		.join(' · ');
}

/**
 * Is a 400 error the template refusing the native effort? Only its own message ("Unexpected
 * reasoning effort ... Supported types are xhigh (default), medium, and low."), nothing broader.
 */
export function isReasoningEffortRejection(message: string): boolean {
	return /unexpected reasoning effort|supported (types|levels)[^.]*xhigh/i.test(message);
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
