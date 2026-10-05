/**
 * Per-conversation tool list snapshot: the tool definitions are the start of every request (they
 * render into the system prompt), so the webui sends a conversation the tool list it started
 * with - a changed list would make the server re-process the whole conversation.
 */

import type { OpenAIToolDefinition } from '$lib/types';

export function parseToolsSnapshot(raw: string | undefined | null): OpenAIToolDefinition[] | null {
	if (!raw) return null;

	try {
		const parsed = JSON.parse(raw);

		return Array.isArray(parsed) ? (parsed as OpenAIToolDefinition[]) : null;
	} catch {
		return null;
	}
}

export interface ToolsChange {
	added: string[];
	removed: string[];
	/** identity of this change, to remember that the user kept the previous tools */
	key: string;
}

/**
 * What the user changed relative to the snapshot: tools enabled now that the snapshot lacks, and
 * snapshot tools that are available but no longer enabled. A snapshot tool that is unavailable
 * (its MCP server is down) is not a change: the conversation keeps it, and a call returns an error.
 */
export function diffToolsSnapshot(
	snapshot: readonly OpenAIToolDefinition[],
	current: readonly OpenAIToolDefinition[],
	available: ReadonlySet<string>
): ToolsChange | null {
	const snap = new Set(snapshot.map((t) => t.function.name));
	const cur = new Set(current.map((t) => t.function.name));
	const added = [...cur].filter((n) => !snap.has(n)).sort();
	const removed = [...snap].filter((n) => !cur.has(n) && available.has(n)).sort();

	if (added.length === 0 && removed.length === 0) return null;

	return { added, key: `+${added.join(',')}|-${removed.join(',')}`, removed };
}

/**
 * The snapshot after applying the user's change: the removed tools go, the added ones come from the
 * current list; everything else - including tools that are unavailable right now - stays frozen.
 */
export function applyToolsChange(
	snapshot: readonly OpenAIToolDefinition[],
	current: readonly OpenAIToolDefinition[],
	change: ToolsChange
): OpenAIToolDefinition[] {
	const removed = new Set(change.removed);
	const added = new Set(change.added);

	return [
		...snapshot.filter((t) => !removed.has(t.function.name)),
		...current.filter((t) => added.has(t.function.name))
	];
}
