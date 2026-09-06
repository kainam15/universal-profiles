ARG BASE_IMAGE
FROM ${BASE_IMAGE}

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        valgrind \
    && rm -rf /var/lib/apt/lists/*

# Compatibility for model images built before the shared runtime was added.
LABEL org.acprof.execution-profile.massif="1"
