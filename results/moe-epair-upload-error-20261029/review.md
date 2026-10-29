# Paired-store upload transport failure and reconciliation

No GPU evaluation occurred in this attempt. The exact e8e3086d candidate was
committed as ff42d15; the controller waited locally for the hourly slot, then
the official CLI upload returned `tls: bad record MAC`. No launch ID was returned.
Controller run012417eff8e244af805643ba84f45c69 logged the error and reverted
to33114871 as d12d820. It is archived here with its unchanged configuration,
event history and full measurement output.

The preserved pre-upload intent is not proof that a job was created. At
05:24:18UTC, authoritative official jobs showed no submission at or after
that intent, and quota matched the pre-upload observation exactly: four
MoE checks used,16left,10scores left. No job/controller remains live from
that failed attempt. The global prior launch remains the04:22:50 Attention
check; one new check is already eligible under the60-minute spacing policy.

`resolution.json` authorizes one unchanged-source transport retry. The original
intent was preserved in this packet and renamed to
`resolved-attempt1-launch-intent.json` in the measurement folder after that
reconciliation. No source, scientific threshold, resource gate or TLS validation
was changed. The next attempt must still check for live official jobs and keep
all quota reserves. Do not repeat an upload after an unknown outcome without
reconciling it again. This is not a parameter sweep or a retry of a GPU failure.

## Independent read-only work during the pacing wait

The source307dd791 Decode preparation draft embeds exactly the same native
MoE CUDA as retained33114871. Applying the frozen candidate's reversible ledger
reproduces CUDA9d6c4354 byte for byte. The old/new patch strings total3516bytes,
so a conditional MAIN-only replacement need not duplicate a full CUDA literal
or add a second MAIN compilation. Small/fallback sources and flags must stay
original, with a distinct new MAIN cache tag and resource admission in prepare.
No Decode source was written or evaluated by this screen.

`decoder-epair-conditional-impact.json` is a sensitivity illustration using
historical D9 benchmark times and a different profiling pass. It assumes the
same MoE speedup transfers, all listed MoE time is exposed/comparable and no
integration overhead. A3% assumed WGS gain yields about0.74% Decode track gain;
7.5% yields about1.80%. These are not forecasts, proven bounds, actual critical
fractions or GPU measurements. A marginal standalone win alone does not justify
spending a Decode check. Whole-layer validation remains necessary for promotion.

Fresh official API snapshots at roughly05:17UTC are preserved in `leaderboards/`:
overall162/1467, MoE66 at2.3483x, Decode162 at1.9940x, Attention104 at2.5624x.
The MoE row binds to retained publicationb6c70c9b83724d2e. Decode's API row binds
to historical441eb580f3074669; Attention's to20524d06e3894595. Those two numbers
must not be assigned to corrected D9 or9e922c35. No withdrawal was authorized.
The account's live overall rank is not a newly established corrected-source rank.

The50th Decode row is2.1580x (job3e3b9fd385ad4af7), about8.43% above D9's
historical1.990224880x check. This does not establish the overall top50 gap.
The raw overall HTTP API still returns only `board` and ranked `rows`.
Some rank ordering aligns with Decode, but ordering is not identical everywhere
and does not prove an aggregation formula. Do not compute a geometric mean
of five tracks or average ranks. Public documentation still describes the
overall endpoint as ranks only; the downloaded official text is preserved.

All verified source and metadata hashes remain unchanged. The active broad
top50 goal remains unmet; continue the authorized main-chat-only work.
