import type { ChatMessageToolCallTiming } from '$lib/types';
import { getContext, setContext } from 'svelte';

/** W2.1: what a tool block's header chip shows (set per tool block by the agentic content). */
export interface ToolChipContext {
	record?: ChatMessageToolCallTiming;
	executingSince: number | null;
	visible: boolean;
}

const TOOL_CHIP_KEY = Symbol.for('llama-ui.tool-chip');

export function setToolChipContext(get: () => ToolChipContext): void {
	setContext(TOOL_CHIP_KEY, get);
}

export function getToolChipContext(): (() => ToolChipContext) | undefined {
	return getContext(TOOL_CHIP_KEY);
}
