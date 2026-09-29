# Environment runtime

The separate roboenv-runtime distribution owns simulator adapters, native device
bindings, world lifecycle, budgets, scoring, and the journaled worker/supervisor.
It is installed in simulator images, separate from agent containers.

The SDK owns the shared task model, device definitions, validation and transport.
Bindings connect typed interfaces to one native world. That world applies the
resolved execution mode before reset: realtime is the default and has no episode
step cap; explicit stepped runs retain configured/native horizons. Agents own
reasoning and controller programs.

Native stepping preserves simulator success/failure rules. Owner submission seals
one verdict, budget and private observation; the Harbor verifier checks that snapshot
and the supervisor journal. Public clients receive instructions and device data.

locks/ records simulator dependency sets; packaging/ contains image relocation and
container recipes. Assets and isolated interpreters are separate build inputs.
Native task exports require an immutable image/depot deployment.
