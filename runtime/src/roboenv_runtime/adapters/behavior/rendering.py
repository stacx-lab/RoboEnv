"""Wait for BEHAVIOR's asynchronous camera buffers without stepping physics."""

from contextlib import contextmanager

ZERO_DELAY_SETTINGS = {
    "/app/renderer/waitIdle": True,
    "/app/hydraEngine/waitIdle": True,
    "/app/updateOrder/checkForHydraRenderComplete": 1000,
}


@contextmanager
def capture_launch_settings():
    """Supply Kit startup arguments through Isaac's documented launcher defaults.

    OmniGibson owns SimulationApp construction and clears sys.argv. Scope these
    defaults to its boot so other app launches retain their original settings.
    """
    from isaacsim import SimulationApp

    previous = SimulationApp.DEFAULT_LAUNCHER_CONFIG
    arguments = [f"--{key}={str(value).lower()}" for key, value in ZERO_DELAY_SETTINGS.items()]
    SimulationApp.DEFAULT_LAUNCHER_CONFIG = {
        **previous, "extra_args": [*previous.get("extra_args", []), *arguments],
    }
    try:
        yield
    finally:
        SimulationApp.DEFAULT_LAUNCHER_CONFIG = previous


def render_for_capture():
    import carb.settings
    import omni.graph.core as graph
    import omnigibson as og

    settings = carb.settings.get_settings()
    if any(settings.get(key) != value for key, value in ZERO_DELAY_SETTINGS.items()):
        raise RuntimeError("BEHAVIOR camera synchronization requires zero-delay startup settings")
    # With zero-delay update ordering a new delivery contains this render's
    # state. A render call alone does not prove that the dispatcher delivered it.
    node = graph.get_node_by_path("/Render/PostProcess/SDGPipeline/PostProcessDispatcher")
    if node is None:
        raise RuntimeError("BEHAVIOR camera dispatcher is unavailable")
    numerator = node.get_attribute("outputs:referenceTimeNumerator")
    denominator = node.get_attribute("outputs:referenceTimeDenominator")

    def stamp():
        value, divisor = numerator.get(), denominator.get()
        return (value, divisor) if value is not None and divisor else None

    previous = stamp()
    for _ in range(16):
        og.sim.render()
        current = stamp()
        if current is not None and current != previous:
            return
    raise RuntimeError("BEHAVIOR cameras did not deliver fresh frames within 16 renders")
