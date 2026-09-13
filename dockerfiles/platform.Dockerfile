# syntax=docker/dockerfile:1
ARG PYTHON_BASE_IMAGE
FROM ${PYTHON_BASE_IMAGE}
ARG PLATFORM_BUILD_FINGERPRINT
ARG PLATFORM_ID
COPY system.lock /opt/acprof/system.lock
RUN --mount=type=bind,source=environment_tools.py,target=/build/environment_tools.py \
    --mount=type=bind,source=dependency_locks.py,target=/build/dependency_locks.py \
    --mount=type=cache,target=/root/.cache/acprof/debs,sharing=locked \
    python /build/environment_tools.py system /opt/acprof/system.lock
COPY requirements.lock expectation.json /opt/acprof/
RUN --mount=type=bind,source=environment_tools.py,target=/build/environment_tools.py \
    --mount=type=bind,source=dependency_locks.py,target=/build/dependency_locks.py \
    --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    python /build/environment_tools.py platform
LABEL org.acprof.platform-build-fingerprint=${PLATFORM_BUILD_FINGERPRINT} \
      org.acprof.platform=${PLATFORM_ID} org.acprof.image-kind="platform"
