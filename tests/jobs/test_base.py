# %% IMPORTS

from agri.io import services
from agri.jobs import base

# %% JOBS


def test_job(
    logger_service: services.LoggerService,
    mlflow_service: services.MlflowService,
) -> None:
    # given
    class MyJob(base.Job):
        KIND: str = "MyJob"

        def run(self) -> base.Locals:
            a, b = 1, "test"
            return locals()

    job = MyJob(logger_service=logger_service, mlflow_service=mlflow_service)
    # when
    with job as runner:
        out = runner.run()
    # then
    assert hasattr(job, "logger_service"), "Job should have a logger service!"
    assert hasattr(job, "mlflow_service"), "Job should have an mlflow service!"
    assert set(out) == {"self", "a", "b"}, "Run should return local variables!"
