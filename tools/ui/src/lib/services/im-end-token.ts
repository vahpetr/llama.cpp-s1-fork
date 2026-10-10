/**
 * W1.2 rule 3: the id of `<|im_end|>` on the server (POST /tokenize with special tokens parsed),
 * cached per model; null when it is not one token or the server cannot say (no bias then).
 */

import { getJsonHeaders } from '$lib/utils/api-headers';

const cache = new Map<string, number | null>();

export async function imEndTokenId(
	model?: string | null,
	fetchImpl: typeof fetch = fetch
): Promise<number | null> {
	const key = model ?? '';

	if (cache.has(key)) return cache.get(key)!;

	let id: number | null = null;

	try {
		const r = await fetchImpl('./tokenize', {
			body: JSON.stringify({
				add_special: false,
				content: '<|im_end|>',
				parse_special: true,
				...(model ? { model } : {})
			}),
			headers: getJsonHeaders(),
			method: 'POST'
		});
		const j = r.ok ? ((await r.json()) as { tokens?: unknown[] }) : undefined;
		const toks = Array.isArray(j?.tokens)
			? j.tokens.map((t) => (typeof t === 'object' && t !== null ? (t as { id?: unknown }).id : t))
			: [];

		id = toks.length === 1 && Number.isInteger(toks[0]) ? (toks[0] as number) : null;
	} catch {
		id = null;
	}

	cache.set(key, id);

	return id;
}

export function clearImEndCacheForTest(): void {
	cache.clear();
}
