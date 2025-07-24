from hatchling.builders.hooks import HookInterface

class BuildHook(HookInterface):
    def initialize(self, version: str):
        self.version = version

    def build(self):
        # Custom build logic can be added here
        pass

if __name__ == "__main__":
    hook = BuildHook()
    hook.initialize("0.1.0")
    hook.build()