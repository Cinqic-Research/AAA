# Prospective Erudition transport successor

The frozen `aaa.erudition.v0` package and its confirmation evidence remain unchanged. Its
`LlamaServer` accepts URL strings by prefix, which permits a URL with loopback text in the
userinfo field to target another host. Python's default redirect handler can also forward
its `Authorization` header after a redirect. These findings block treating that transport
as a secure current runtime interface.

`research.aaa_erudition_v1.transport.LoopbackRuntime` is a prospective replacement. It
accepts only `http://127.0.0.1:<port>` or `http://[::1]:<port>` with an explicit port and no
userinfo, path, query or fragment. It disables environment proxies and refuses every
redirect. Its tests verify that a misleading URL is rejected, a redirect cannot deliver
the bearer header to a second listener, and a valid local request reaches the selected
listener.

This module is not wired into the frozen phase, and no prior result is recomputed or
reinterpreted through it. A successor research protocol still needs a versioned source
identity, integration into its entry points, exact-head validation, runtime listener
ownership checks where applicable, and fresh evidence before any scientific promotion.
