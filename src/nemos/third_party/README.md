# Temporary vendoring of JAXopt

NeMoS currently relies on [JAXopt](https://github.com/google/jaxopt) as its optimization backend.
\
As JAXopt is no longer maintained, we are in the process of transitioning to [Optimistix](https://github.com/patrick-kidger/optimistix) and [Optax](https://github.com/google-deepmind/optax).
\
In order to be able to issue quick fixes in case new JAX versions break JAXopt's functionality,
we temporarily vendor JAXopt's last release (0.8.5) until the transition is successful.

## Changes made to `JAXopt`
- Removed docs, examples, benchmarks
- Changed import paths to import `jaxopt` from `nemos.third_party`
- Removed Github workflows (`.github`)
