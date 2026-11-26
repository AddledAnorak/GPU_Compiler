"""Typed host-driver failures, independent of a particular transport."""


class DriverError(Exception):
    pass


class ValidationError(DriverError, ValueError):
    pass


class ProgramFormatError(ValidationError):
    pass


class ResourceError(DriverError):
    pass


class ResourceBusyError(ResourceError):
    pass


class DriverClosedError(DriverError):
    pass


class SubmissionError(DriverError):
    pass


class ExecutionTimeoutError(SubmissionError, TimeoutError):
    pass


class FenceTimeoutError(DriverError, TimeoutError):
    pass


class SubmissionCancelledError(SubmissionError):
    pass


class HardwareSafetyError(DriverError):
    pass


class HardwareBusyError(HardwareSafetyError):
    pass
