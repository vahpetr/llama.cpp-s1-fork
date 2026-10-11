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
import { chatStore } from '$lib/stores/chat/index.svelte';
import { modelsStore } from '$lib/stores/models/index.svelte';
import { serverStore } from '$lib/stores/server.svelte';
import { settingsStore } from '$lib/stores/settings/index.svelte';
import {
	BACKEND_SAMPLING_OLD_SERVER_NOTE,
	BACKEND_SAMPLING_SPECULATIVE_NOTE,
	backendSamplingHelp,
	backendSamplingHint,
	backendSamplingNote,
	backendSamplingState,
	exportWithBackendSamplingLock,
	isBackendSamplingRejection
} from '$lib/utils/backend-sampling';

afterEach(() => {
	vi.restoreAllMocks();
	vi.unstubAllGlobals();
	modelsStore.props.backendSamplingRejected.clear();
	serverStore.props = null;
	settingsStore.config = { ...settingsStore.config, backend_sampling: false };
});

describe('backend sampling availability from /props', () => {
	it('locked with the reason the server reports', () => {
		const mtp = backendSamplingState(
			{ backend_sampling_supported: false, speculative: { n_max: 4, type: 'draft-mtp' } },
			false
		);

		expect(backendSamplingHint(mtp)).toBe(
			'Not available on this server: a speculative verify round needs 1+4 outputs per sequence, more than the context allows'
		);
		expect(
			backendSamplingHint(
				backendSamplingState(
					{ backend_sampling_supported: false, speculative: { type: 'none' } },
					false
				)
			)
		).toBe('Not available on this server: tensor split');
		expect(backendSamplingState({ backend_sampling_supported: true }, false)).toEqual({
			supported: true
		});
	});

	it('supported under draft-mtp: enabled, with the cost note (no lock)', () => {
		const mtp = backendSamplingState(
			{ backend_sampling_supported: true, speculative: { n_max: 4, type: 'draft-mtp' } },
			false
		);

		expect(backendSamplingHint(mtp)).toBeNull();
		expect(backendSamplingNote(mtp)).toBe(BACKEND_SAMPLING_SPECULATIVE_NOTE);
		expect(
			backendSamplingNote(
				backendSamplingState(
					{ backend_sampling_supported: true, speculative: { type: 'none' } },
					false
				)
			)
		).toBeNull();
	});

	it('an older server (no field): enabled, the description warns', () => {
		const old = backendSamplingState({}, false);

		expect(old).toEqual({ supported: undefined });
		expect(backendSamplingHint(old)).toBeNull();
		expect(backendSamplingHelp('Enable backend-based samplers.', old)).toBe(
			`Enable backend-based samplers. ${BACKEND_SAMPLING_OLD_SERVER_NOTE}`
		);
		expect(backendSamplingHelp('x', { supported: true })).toBe('x');
	});
});

describe('reactive guard', () => {
	it('only the sampling-specific text counts (a generic 500 does not), only with it on', () => {
		expect(
			isBackendSamplingRejection(
				'backend_sampling is not supported on this server: tensor split',
				true
			)
		).toBe(true);
		expect(
			isBackendSamplingRejection('backend_sampling is not supported on this server', false)
		).toBe(false);
		expect(isBackendSamplingRejection('Invalid input batch.', true)).toBe(false);
		expect(isBackendSamplingRejection('some other bad request', true)).toBe(false);
	});

	async function send(status: number, message: string) {
		vi.spyOn(serverStore, 'isRouterMode', 'get').mockReturnValue(false);
		serverStore.props = { model_alias: 'brain' } as never;
		settingsStore.config = { ...settingsStore.config, backend_sampling: true };

		const bodies: Record<string, unknown>[] = [];

		vi.stubGlobal(
			'fetch',
			vi.fn(async (_u: string, init?: RequestInit) => {
				bodies.push(JSON.parse(String(init?.body)));

				return new Response(JSON.stringify({ error: { code: status, message } }), { status });
			})
		);
		await ChatService.sendMessage([{ content: 'x', role: 'user' }], {
			...chatStore.getApiOptions(),
			stream: false
		} as never).catch(() => {});

		return bodies;
	}

	it('the 400 rejection unchecks the setting and remembers the server', async () => {
		const bodies = await send(
			400,
			'backend_sampling is not supported on this server: tensor split'
		);

		expect(bodies[0].backend_sampling).toBe(true);
		expect(settingsStore.config.backend_sampling).toBe(false);
		expect([...modelsStore.props.backendSamplingRejected.keys()]).toEqual(['brain']);
		expect(modelsStore.props.backendSampling().supported).toBe(false);
	});

	it('a generic 500 "Invalid input batch." is ignored', async () => {
		await send(500, 'Invalid input batch.');

		expect(settingsStore.config.backend_sampling).toBe(true);
		expect(modelsStore.props.backendSamplingRejected.size).toBe(0);
	});

	it('an error event inside the SSE stream triggers the guard too; a /props refresh clears the memory', async () => {
		vi.spyOn(serverStore, 'isRouterMode', 'get').mockReturnValue(false);
		serverStore.props = { model_alias: 'brain' } as never;
		settingsStore.config = { ...settingsStore.config, backend_sampling: true };
		vi.stubGlobal(
			'fetch',
			vi.fn(
				async () =>
					new Response(
						`data: ${JSON.stringify({ error: { code: 400, message: 'backend_sampling is not supported on this server: tensor split' } })}\n\ndata: [DONE]\n\n`,
						{ headers: { 'Content-Type': 'text/event-stream' } }
					)
			)
		);
		await ChatService.sendMessage([{ content: 'x', role: 'user' }], {
			...chatStore.getApiOptions(),
			stream: true
		} as never).catch(() => {});

		expect(settingsStore.config.backend_sampling).toBe(false);
		expect(modelsStore.props.backendSampling().supported).toBe(false);

		// remembered: a re-enabled setting is not sent to this server
		settingsStore.config = { ...settingsStore.config, backend_sampling: true };
		expect(chatStore.getApiOptions().backend_sampling).toBeUndefined();

		// a refreshed /props (another object) forgets it
		serverStore.props = { model_alias: 'brain' } as never;
		expect(modelsStore.props.backendSampling().supported).toBeUndefined();
		expect(chatStore.getApiOptions().backend_sampling).toBe(true);
	});

	it('the capability holds on the final body: the custom JSON cannot re-add backend_sampling', async () => {
		vi.spyOn(serverStore, 'isRouterMode', 'get').mockReturnValue(false);
		serverStore.props = {
			backend_sampling_supported: false,
			speculative: { type: 'none' }
		} as never;
		settingsStore.config = {
			...settingsStore.config,
			backend_sampling: false,
			customJson: '{"backend_sampling":true}'
		};

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

		expect(bodies[0].backend_sampling).toBeUndefined();
		settingsStore.config = { ...settingsStore.config, customJson: '' };
	});

	it('a locked setting is exported as false; nothing else changes', () => {
		const data = { config: { backend_sampling: true, temperature: 0.5 }, version: 1 };

		expect(
			exportWithBackendSamplingLock(data, { reason: 'tensor split', supported: false }).config
		).toEqual({ backend_sampling: false, temperature: 0.5 });
		expect(exportWithBackendSamplingLock(data, { supported: true })).toBe(data);
		expect(exportWithBackendSamplingLock(data, { supported: undefined })).toBe(data);
	});

	it('a server reporting it unsupported never gets backend_sampling', () => {
		vi.spyOn(serverStore, 'isRouterMode', 'get').mockReturnValue(false);
		serverStore.props = {
			backend_sampling_supported: false,
			speculative: { type: 'draft-mtp' }
		} as never;
		settingsStore.config = { ...settingsStore.config, backend_sampling: true };
		expect(chatStore.getApiOptions().backend_sampling).toBeUndefined();
	});
});
