class FirstAvailableAllocator:
    def choose_action(self, available_actions):
        """
        Allocate the first available parking slot.
        """
        return min(available_actions)
