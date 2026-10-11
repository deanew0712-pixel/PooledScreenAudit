FROM python:3.11-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
WORKDIR /opt/psa
COPY . /opt/psa
RUN python -m pip install --no-cache-dir . && python -m unittest discover -s tests -v && python tests/numerical_fixture.py
RUN useradd --create-home --uid 10001 psa && mkdir /work && chown psa /work
USER psa
WORKDIR /work
ENTRYPOINT ["pooledscreenaudit"]
CMD ["--help"]
