/**
 * Backend sampling availability (contrib 0225; the server side is 0226): a server that cannot run
 * samplers on the backend - speculative decoding verifies more than one output per sequence, or a
 * tensor split - reports `/props.backend_sampling_supported: false` (+ `speculative.type`) and
 * rejects `backend_sampling: true` with a 400; older servers do not report it and may fail with
 * HTTP 500 "Invalid input batch." (clearing the slot cache).
 */

export interface BackendSamplingProps {
	backend_sampling_supported?: boolean;
	speculative?: { type?: string; n_max?: number } | boolean;
}

export interface BackendSamplingState {
	/** true / false as the server reports it (or as a rejection taught us); undefined = not reported */
	supported: boolean | undefined;
	/** why it is unavailable: "speculative decoding (draft-mtp)" / "tensor split" / ... */
	reason?: string;
}

export const BACKEND_SAMPLING_OLD_SERVER_NOTE =
	'May fail with speculative decoding (draft-mtp) and clear the slot cache on older servers.';

export function backendSamplingState(
	props: BackendSamplingProps | null | undefined,
	rejected: boolean
): BackendSamplingState {
	const spec = props?.speculative;
	const type = spec && typeof spec === 'object' ? spec.type : undefined;
	const reason = type && type !== 'none' ? `speculative decoding (${type})` : 'tensor split';

	if (props?.backend_sampling_supported === false) return { reason, supported: false };

	if (rejected) {
		return {
			reason: type && type !== 'none' ? reason : 'rejected by the server',
			supported: false
		};
	}

	if (props?.backend_sampling_supported === true) return { supported: true };

	return { supported: undefined };
}

/** The Settings hint for a locked checkbox. */
export function backendSamplingHint(state: BackendSamplingState): string | null {
	return state.supported === false ? `Not available on this server: ${state.reason}` : null;
}

/** The Settings description: the field's help, plus the old-server note when nothing is reported. */
export function backendSamplingHelp(help: string, state: BackendSamplingState): string {
	return state.supported === undefined ? `${help} ${BACKEND_SAMPLING_OLD_SERVER_NOTE}` : help;
}

/**
 * Is a failed request the server refusing backend sampling? The new 400 text, or an older
 * server's HTTP 500 "Invalid input batch." while backend sampling was on.
 */
export function isBackendSamplingRejection(
	status: number,
	message: string,
	backendSamplingOn: boolean
): boolean {
	if (!backendSamplingOn) return false;

	if (status === 400 && /backend_sampling is not supported on this server/i.test(message)) {
		return true;
	}

	return status === 500 && /invalid input batch/i.test(message);
}
