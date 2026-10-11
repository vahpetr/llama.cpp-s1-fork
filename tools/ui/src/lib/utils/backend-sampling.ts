/**
 * Backend sampling availability (contrib 0225b; the server side is 0226): a server that cannot run
 * samplers on the backend (a tensor split, or a real limit mismatch) reports
 * `/props.backend_sampling_supported: false` (+ `speculative.type`) and rejects
 * `backend_sampling: true` with a 400 naming it. Under draft-mtp the server supports it (the limit
 * is 1 + n_max on the target) - enabled, with a cost note. Older servers do not report it.
 */

export interface BackendSamplingProps {
	backend_sampling_supported?: boolean;
	speculative?: { type?: string; n_max?: number } | boolean;
}

export interface BackendSamplingState {
	/** true / false as the server reports it (or as a rejection taught us); undefined = not reported */
	supported: boolean | undefined;
	/** why it is unavailable: "a speculative verify round needs 1+N outputs ..." / "tensor split" / ... */
	reason?: string;
	/** the server's speculative decoding type ("draft-mtp", ...; undefined / "none" = none) */
	speculativeType?: string;
}

export const BACKEND_SAMPLING_SPECULATIVE_NOTE =
	'Works with speculative decoding on this server, but bypasses the graph cache and the CPU fast path (slower decode) and is not bit-exact at temperature > 0.';

export const BACKEND_SAMPLING_OLD_SERVER_NOTE =
	'May fail with speculative decoding (draft-mtp) and clear the slot cache on older servers.';

export function backendSamplingState(
	props: BackendSamplingProps | null | undefined,
	rejected: boolean
): BackendSamplingState {
	const spec = props?.speculative;
	const type = spec && typeof spec === 'object' ? spec.type : undefined;
	const nMax = spec && typeof spec === 'object' ? spec.n_max : undefined;
	// the server's limit reason (0226b): a verify round needs 1 + n_max outputs per sequence, more
	// than the target context allows; without speculative decoding it is a tensor split
	const reason =
		type && type !== 'none'
			? `a speculative verify round needs 1+${nMax ?? 'N'} outputs per sequence, more than the context allows`
			: 'tensor split';

	if (props?.backend_sampling_supported === false) return { reason, supported: false };

	if (rejected) {
		return {
			reason: type && type !== 'none' ? reason : 'rejected by the server',
			supported: false
		};
	}

	if (props?.backend_sampling_supported === true) {
		return type && type !== 'none'
			? { speculativeType: type, supported: true }
			: { supported: true };
	}

	return { supported: undefined };
}

/** The Settings hint for a locked checkbox. */
export function backendSamplingHint(state: BackendSamplingState): string | null {
	return state.supported === false ? `Not available on this server: ${state.reason}` : null;
}

/** The informational note under an enabled checkbox on a server with speculative decoding. */
export function backendSamplingNote(state: BackendSamplingState): string | null {
	return state.supported === true && state.speculativeType
		? BACKEND_SAMPLING_SPECULATIVE_NOTE
		: null;
}

/** The Settings description: the field's help, plus the old-server note when nothing is reported. */
export function backendSamplingHelp(help: string, state: BackendSamplingState): string {
	return state.supported === undefined ? `${help} ${BACKEND_SAMPLING_OLD_SERVER_NOTE}` : help;
}

/**
 * Is an error the server refusing backend sampling? Only its sampling-specific text
 * ("backend_sampling is not supported on this server ...", a 400 or an error event in the stream)
 * while backend sampling was on - nothing generic (e.g. a 500 "Invalid input batch.").
 */
export function isBackendSamplingRejection(message: string, backendSamplingOn: boolean): boolean {
	return backendSamplingOn && /backend_sampling is not supported on this server/i.test(message);
}

/**
 * A settings export with backend sampling written as false while it is locked on this server (no
 * underlying true travels); the rejection memory is runtime-only and never exported.
 */
export function exportWithBackendSamplingLock<T extends { config: Record<string, unknown> }>(
	data: T,
	state: BackendSamplingState
): T {
	if (state.supported !== false || data.config.backend_sampling === undefined) return data;

	return { ...data, config: { ...data.config, backend_sampling: false } };
}
