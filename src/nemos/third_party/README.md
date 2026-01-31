# Vendored libraries

We may temporarily vendor third-party libraries, meaning we include license-compatible external libraries directly within NeMoS's source code.
This is only done in rare cases, such as when an upstream dependency is deprecated or becomes unmaintained.
Vendoring is always intended as a short-term solution to provide stability while we migrate away from the dependency.
Our goal is to remove vendored code as soon as a suitable long-term alternative is available.

Our approach here is inspired by how [pip](https://github.com/pypa/pip/tree/main/src/pip/_vendor) vendors third-party libraries.

Changes made to vendored libraries should be documented below.

# Temporary vendoring of JAXopt

Early versions of NeMoS used to rely on [JAXopt](https://github.com/google/jaxopt) as its optimization backend.
\
As JAXopt is no longer maintained, the backend was migrated to [Optimistix](https://github.com/patrick-kidger/optimistix) and [Optax](https://github.com/google-deepmind/optax).
\
In order to be able to issue quick fixes in case new JAX versions break JAXopt's functionality,
JAXopt's last release (0.8.5) was vendored until the transition was successful (January, 2026).
