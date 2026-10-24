"""Policy-visible prompt projection shared by Tau2/Tau3 data and runtime.

The official AgentGym LLM agent wraps the domain policy in a fixed system
instruction.  Historical trajectory import and live RL transport must use the
same bytes; keeping the projection here prevents the two stages from silently
drifting.
"""

from __future__ import annotations


TAU2_AGENT_INSTRUCTION = """
You are a customer service agent that helps the user according to the <policy> provided below.
In each turn you can either:
- Send a message to the user.
- Make a tool call.
You cannot do both at the same time.

Try to be helpful and always follow the policy. Always make sure you generate valid JSON only.
""".strip()

TAU2_SYSTEM_PROMPT = """
<instructions>
{agent_instruction}
</instructions>
<policy>
{domain_policy}
</policy>
""".strip()


def tau2_agent_system_prompt(domain_policy: str) -> str:
    """Return the exact system message used by the official LLM agent."""

    return TAU2_SYSTEM_PROMPT.format(
        agent_instruction=TAU2_AGENT_INSTRUCTION,
        domain_policy=str(domain_policy),
    )
