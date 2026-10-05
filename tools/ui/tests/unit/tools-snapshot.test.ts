import { applyToolsChange, diffToolsSnapshot, parseToolsSnapshot } from '$lib/utils/tools-snapshot';
import { describe, expect, it } from 'vitest';

const t = (name: string) =>
	({ function: { description: name, name, parameters: {} }, type: 'function' }) as never;

describe('tool list snapshot', () => {
	it('no change when the lists match', () => {
		expect(diffToolsSnapshot([t('a'), t('b')], [t('b'), t('a')], new Set(['a', 'b']))).toBeNull();
	});

	it('an unavailable snapshot tool (server down) is not a change', () => {
		expect(diffToolsSnapshot([t('a'), t('mcp_x')], [t('a')], new Set(['a']))).toBeNull();
	});

	it('reports tools the user enabled or disabled, with a stable key', () => {
		const change = diffToolsSnapshot([t('a'), t('b')], [t('a'), t('c')], new Set(['a', 'b', 'c']));

		expect(change).toEqual({ added: ['c'], key: '+c|-b', removed: ['b'] });
	});

	it('applies a change as a delta: an unavailable frozen tool stays', () => {
		const snapshot = [t('a'), t('b_offline')];
		const current = [t('c')];
		const change = diffToolsSnapshot(snapshot, current, new Set(['a', 'c']))!;
		const next = applyToolsChange(snapshot, current, change).map(
			(x: { function: { name: string } }) => x.function.name
		);

		expect(change).toEqual({ added: ['c'], key: '+c|-a', removed: ['a'] });
		expect(next).toEqual(['b_offline', 'c']);
	});

	it('an empty frozen list still reports enabled tools as a change', () => {
		expect(diffToolsSnapshot([], [t('a')], new Set(['a']))?.added).toEqual(['a']);
	});

	it('parses only arrays', () => {
		expect(parseToolsSnapshot('[{"type":"function"}]')?.length).toBe(1);
		expect(parseToolsSnapshot('{}')).toBeNull();
		expect(parseToolsSnapshot('oops')).toBeNull();
		expect(parseToolsSnapshot(undefined)).toBeNull();
	});
});
