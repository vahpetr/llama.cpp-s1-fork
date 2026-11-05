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

import { ChatService } from '$lib/services/chat.service';
import { serverStore } from '$lib/stores/server.svelte';

afterEach(() => {
	vi.restoreAllMocks();
	vi.unstubAllGlobals();
});

/** A mocked server reply: SSE chunks (stream) or one JSON body. */
function sse(chunks: unknown[]): Response {
	const text = chunks.map((c) => `data: ${JSON.stringify(c)}\n\n`).join('') + 'data: [DONE]\n\n';

	return new Response(text, { headers: { 'content-type': 'text/event-stream' }, status: 200 });
}

async function complete(response: Response, stream: boolean) {
	vi.spyOn(serverStore, 'isRouterMode', 'get').mockReturnValue(false);
	vi.stubGlobal(
		'fetch',
		vi.fn(async () => response)
	);

	let got: { content?: string; finish?: string; stopType?: string } = {};

	await ChatService.sendMessage([{ content: 'x', role: 'user' }], {
		onComplete: (
			c: string,
			_r?: string,
			_t?: unknown,
			_tc?: string,
			finish?: string,
			st?: string
		) => {
			got = { content: c, finish, stopType: st };
		},
		stream
	} as never);

	return got;
}

describe('stop_type next to finish_reason (drift guard contract, mocked server)', () => {
	it('stream: the final chunk carries finish_reason "stop" + stop_type "think_reenter"', async () => {
		const got = await complete(
			sse([
				{ choices: [{ delta: { content: 'Ответ. ' } }] },
				{ choices: [{ delta: { content: 'Let me' } }] },
				{ choices: [{ delta: {}, finish_reason: 'stop', stop_type: 'think_reenter' }] }
			]),
			true
		);

		expect(got).toEqual({ content: 'Ответ. Let me', finish: 'stop', stopType: 'think_reenter' });
	});

	it('stream: an ordinary stop has no stop_type', async () => {
		const got = await complete(
			sse([
				{ choices: [{ delta: { content: 'done.' } }] },
				{ choices: [{ delta: {}, finish_reason: 'stop' }] }
			]),
			true
		);

		expect(got.finish).toBe('stop');
		expect(got.stopType).toBeUndefined();
	});

	it('non-stream: stop_type is passed on, finish_reason stays unreported (rules 2 / 3 never ran)', async () => {
		const got = await complete(
			new Response(
				JSON.stringify({
					choices: [
						{
							finish_reason: 'stop',
							message: { content: 'partial', role: 'assistant' },
							stop_type: 'think_reenter'
						}
					]
				}),
				{ status: 200 }
			),
			false
		);

		expect(got).toEqual({ content: 'partial', finish: undefined, stopType: 'think_reenter' });
	});
});
