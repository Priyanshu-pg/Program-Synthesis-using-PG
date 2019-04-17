import brainfuck as bf
from collections import namedtuple

RewardInfo = namedtuple('RewardInfo', ['episode_rewards', 'input_case',
                                       'correct_output',
                                       'code_output', 'reason', 'input_type',
                                       'output_type'])


class IOTuple(tuple):
  pass


def abs_diff(a, b, base=0):
  """Absolute value of difference between scalars.

  abs_diff is symmetric, i.e. `a` and `b` are interchangeable.

  Args:
    a: First argument. An int.
    b: Seconds argument. An int.
    base: Dummy argument so that the argument signature matches other scalar
        diff functions. abs_diff is the same in all bases.

  Returns:
    abs(a - b).
  """
  del base  # Unused.
  return abs(a - b)


def absolute_distance(pred, target, base, scalar_diff_fn=abs_diff):
  """Asymmetric list distance function.

  List distance is the sum of element-wise distances, like Hamming distance, but
  where `pred` can be longer or shorter than `target`. For each position in both
  `pred` and `target`, distance between those elements is computed with
  `scalar_diff_fn`. For missing or extra elements in `pred`, the maximum
  distance is assigned, which is equal to `base`.

  Distance is 0 when `pred` and `target` are identical, and will be a positive
  integer when they are not.

  Args:
    pred: Prediction list. Distance from this list is computed.
    target: Target list. Distance to this list is computed.
    base: The integer base to use. For example, a list of chars would use base
        256.
    scalar_diff_fn: Element-wise distance function.

  Returns:
    List distance between `pred` and `target`.
  """
  d = 0
  for i, target_t in enumerate(target):
    if i >= len(pred):
      d += base  # A missing slot is worth the max distance.
    else:
      # Add element-wise distance for this slot.
      d += scalar_diff_fn(pred[i], target_t, base)
  if len(pred) > len(target):
    # Each extra slot is worth the max distance.
    d += (len(pred) - len(target)) * base
  return d


def absolute_distance_reward(pred, target, base, scalar_diff_fn=abs_diff):
  """Reward function based on absolute_distance function.

  Maximum reward, 1.0, is given when the lists are equal. Reward is scaled
  so that 0.0 reward is given when `pred` is the empty list (assuming `target`
  is not empty). Reward can go negative when `pred` is longer than `target`.

  This is an asymmetric reward function, so which list is the prediction and
  which is the target matters.

  Args:
    pred: Prediction sequence. This should be the sequence outputted by the
        generated code. List of ints n, where 0 <= n < base.
    target: Target sequence. The correct sequence that the generated code needs
        to output. List of ints n, where 0 <= n < base.
    base: Base of the computation.
    scalar_diff_fn: Element-wise distance function.

  Returns:
    Reward computed based on `pred` and `target`. A float.
  """
  unit_dist = float(base * len(target))
  if unit_dist == 0:
    unit_dist = base
  dist = absolute_distance(pred, target, base, scalar_diff_fn=scalar_diff_fn)
  return (unit_dist - dist) / unit_dist


def clipped_linear(x, x0, y0, slope, y_range):
  min_y, max_y = y_range
  return min(max(slope * (x - x0) + y0, min_y), max_y)


class IOType(object):
  string = 'string'
  integer = 'integer'
  boolean = 'boolean'


class PrintTask():
    """Print string coding task.

      Code needs to output a fixed string (given as a hyperparameter to the
      task constructor). Program input is ignored.
      """

    def __init__(self, base=256, fixed_string=None):
        self.base = base  # base includes EOS
        self.eos = 0
        if fixed_string:
            self.fixed_string = fixed_string
        else:
            self.fixed_string = [0, 1, 2, 27]  # ABC<EOS>
        self.min_length = self.max_length = len(self.fixed_string)

    def make_io_set(self):
        return [(list(), list(self.fixed_string))]


def compute_best_reward(task, reward_fn, correct_bonus, code_length_bonus):
    io_seqs = task.make_io_set()
    reward = 0.0
    for _, output_seq in io_seqs:
        reward += reward_fn(output_seq, output_seq, task.base)
        reward += correct_bonus
        reward += code_length_bonus  # Bonus for shortest code.
    best_reward = reward
    return best_reward
    # self.good_reward = 0.75 * reward


def get_reward(code_string):

    task = PrintTask(base=27, fixed_string=[7, 4, 11, 11, 14]) # print hello

    max_execution_steps = 5000
    require_correct_syntax = False
    io_seqs = task.make_io_set()
    terminal_reward = 0.0
    failure_reward = -2.0
    correct_bonus = 1.0
    max_code_length = 32
    min_code_length = 0
    code_length_bonus = 1.0
    reward_fn = absolute_distance_reward
    time_penalty = (
        1.0 / (max_code_length - min_code_length)
        if max_code_length > min_code_length else 0.0)
    input_type = (
        task.input_type if hasattr(task, 'input_type') else IOType.integer)
    output_type = (
        task.output_type if hasattr(task, 'output_type')
        else IOType.integer)
    best_reward = compute_best_reward(task, reward_fn, correct_bonus, code_length_bonus)

    results = []
    reason = 'correct'
    for input_seq, output_seq in io_seqs:
      eval_result = bf.evaluate(
          code_string[:-1], input_buffer=input_seq, timeout=0.1, #not send EOS to interpreter
          max_steps=max_execution_steps,
          base=task.base,
          require_correct_syntax=require_correct_syntax)
      result, success = eval_result.output, eval_result.success
      if not success:
          # Code execution timed out.
          terminal_reward = failure_reward
          results = []
          reason = eval_result.failure_reason
          break
      else:
          terminal_reward += reward_fn(result, output_seq, task.base)
          if result == output_seq:
              terminal_reward += correct_bonus  # Bonus for correct answer.
              # Only add additional reward for shorter code. Subtracting reward
              # interferes with the main objective. Only optimize for length once
              # any solution is found.
              if min_code_length == max_code_length:
                  terminal_reward += code_length_bonus
              else:
                  terminal_reward += code_length_bonus * clipped_linear(
                      x=len(code_string), x0=min_code_length, y0=1.0,
                      slope=-time_penalty, y_range=(0.0, 1.0))

              # reason remains 'correct' if it is already
          elif reason == 'correct':
              reason = 'wrong'
      results.append(result)

    # Return list of rewards, one for each char in the code. All are 0 except
    # for the terminal reward.
    terminal_reward /= best_reward
    return RewardInfo(
        episode_rewards=[0.0] * (len(code_string) - 1) + [terminal_reward],
        input_case=IOTuple(i for i, o in io_seqs),
        correct_output=IOTuple(o for i, o in io_seqs),
        code_output=IOTuple(results),
        input_type=input_type,
        output_type=output_type,
        reason=reason)
