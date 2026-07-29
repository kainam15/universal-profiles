ARG BASE_IMAGE
FROM ${BASE_IMAGE}

# Nsys itself is mounted from the host, but QdstrmImporter dynamically links
# against libdw.so.1.  Debian 13 renamed the package for the 64-bit time_t ABI;
# older Debian/Ubuntu bases still expose it as libdw1.
RUN apt-get update \
    && if apt-cache show libdw1t64 >/dev/null 2>&1; then \
         elfutils_runtime=libdw1t64; \
       else \
         elfutils_runtime=libdw1; \
       fi \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
         "${elfutils_runtime}" \
    && rm -rf /var/lib/apt/lists/*
