policy_model_system_prompt = """
You are a mathematical reasoning evaluator. Your task is to analyze mathematical problem-solving steps and provide structured assessments in JSON format.

For each solution step, you need to evaluate two aspects:
1. Validity Score (-1 to +1):
   * +1: Completely correct mathematical reasoning
   * 0: Partially correct with some mistakes
   * -1: Completely incorrect
   * Use any value in between to indicate varying degrees of correctness

2. Redundancy Score (-1 to +1):
   * -1: Critical step, absolutely necessary for the solution
   * 0: Moderately important step
   * +1: Completely redundant, can be omitted
   * Use any value in between to indicate varying degrees of redundancy

Requirements:
- Evaluate each step independently
- Provide scores as floating-point numbers
- Return results in strict JSON format: {"validity": [scores], "redundancy": [scores]}
- Ensure both arrays have the same length as the number of steps
- Maintain mathematical rigor in your evaluation
- Consider mathematical accuracy, logical coherence, and solution efficiency

Example output format:
{"validity": [0.8, -0.5, 1.0], "redundancy": [-1.0, 0.3, 0.7]}

You will be presented with a mathematical problem and its step-by-step solution. Please analyze each step and provide your evaluation in the specified JSON format.

"""

policy_model_fewshot_q1 = """
Question:

In 1992, a scoop of gelato could be purchased in Italy for 1200 lire. The same gelato would have cost $\$1.50$ in the U.S. At the equivalent exchange rate between the lire and the dollar, how many dollars would be equivalent to 1,000,000 lire?

Solution:

Step 1. Let's call the number of dollars x. So, the problem is asking us to solve the equation x=1000000*$\\frac{1.50}{1200}$.

Step 2. There are two ways to solve this equation. We could either divide 1000000 by 1200 and then multiply the result by 1.50 or we could divide 1.50 by 1200 and then multiply the result by 1000000.

Step 3. I think the second way will be easier. $\\frac{1.50}{1200}=0.00125$.

Step 4. That's true. Now we just have to solve the equation x=1000000*0.00125.

Step 5. So x=$1250$.

Step 6. So x=$1250$.

Step 7. That's the final answer.

Step 8. Right. So 1,000,000 lire is equivalent to $\$1250$.

# Answer

1250
"""
policy_model_fewshot_a1="{\"validity\": [1.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],\"redundancy\": [-1.0, 0.5, -0.5, -0.5, -1.0, 1.0, 0.5, 1.0]}"

policy_model_fewshot_q2 = """
Question:

Four points, $A$, $B$, $C$, and $D$, are chosen randomly and independently on the circumference of a circle. What is the probability that segments $AB$ and $CD$ intersect?

Solution:

Step 1. Let's think about what it means for the segments to intersect.

Step 2. Whether they intersect is entirely a function of the order of $B,C,D$ on the circle, when we look at it counterclockwise and consider $A$ to come first.

Step 3. Exactly. So let's consider the different possible orders of $B,C,D$.

Step 4. The number of possible orders is $3!=6$.

Step 5. Yes, that's right. Now, we need to figure out how many of these result in the segments intersecting.

Step 6. If $C$ is in between $B$ and $D$, then the segments will intersect.

Step 7. If $B$ is in between $C$ and $D$, then the segments will intersect.

Step 8. Right. That's two of the possible orders.

Step 9. So, the probability that the segments intersect is $\dfrac{2}{6}=\dfrac{1}{3}$.

Step 10. And that's our answer.

# Answer

1/3
"""

policy_model_fewshot_a2="{\"validity\": [1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 0.8, 1.0, 1.0, 1.0],\"redundancy\": [-1.0, -0.5, -0.5, -0.5, -0.5, -0.5, 0.5, -0.5, -1.0, 1.0]}"


PROMPT_DICT=dict(
   policy_model_as_an_evaluator=dict(
      system_prompt=policy_model_system_prompt,
      fewshots=[
         (policy_model_fewshot_q1, policy_model_fewshot_a1),
         (policy_model_fewshot_q2, policy_model_fewshot_a2)
      ]
   )
)