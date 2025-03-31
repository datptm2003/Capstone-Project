import torch
from transformers import AutoTokenizer, BitsAndBytesConfig
from trl import AutoModelForCausalLMWithValueHead
from app.services import llm_service  # External LLM service
from rw_model.predictor import ActionEvaluator  # Reward model
from prompts import rl_prompt  # Assuming this exists from training code

# Configurations
MODEL_DIR = "./other_sft/checkpoint"  # Pre-trained SFT model directory
REWARD_MODEL_DIR = "./rw_model/checkpoint"  # Reward model checkpoint
OUTPUT_MODEL_DIR = "./rlhf_model"  # RLHF fine-tuned model directory
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
REWARD_THRESHOLD = 0.8  # Threshold for "good enough" action sequence
MAX_ITERATIONS = 5  # Maximum refinement iterations
BATCH_SIZE = 4  # Number of scenes/goals to process at once

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, padding_side="left")
tokenizer.pad_token = tokenizer.eos_token

# Quantization config for 4-bit
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

# Load RLHF policy model with quantization
policy_model = AutoModelForCausalLMWithValueHead.from_pretrained(
    MODEL_DIR,
    quantization_config=bnb_config,
    device_map=DEVICE
)
policy_model.eval()  # Set to evaluation mode

# Load reward model
reward_model = ActionEvaluator().to(DEVICE)
reward_model.load_state_dict(torch.load(f"{REWARD_MODEL_DIR}/action_evaluator.pth"))
reward_model.eval()

def generate_feedback_batch(scenes, goals, actions_batch):
    """Generate feedback for a batch using the RLHF policy model."""
    # Format prompts for the batch
    prompts = [rl_prompt.format(scene=s, goal=g, actions=a) for s, g, a in zip(scenes, goals, actions_batch)]
    encoded_prompts = tokenizer(prompts, return_tensors="pt", truncation=True, padding=True, max_length=512)
    query_tensor = encoded_prompts.input_ids.to(DEVICE)
    attention_mask = encoded_prompts.attention_mask.to(DEVICE)

    # Generate feedback for the batch
    with torch.no_grad():
        response_tensors = policy_model.generate(
            query_tensor,
            attention_mask=attention_mask,
            max_length=256,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,  # Enable sampling for varied feedback
            temperature=0.7,
        )
    
    # Decode feedback for each item in the batch
    feedback_batch = [tokenizer.decode(response, skip_special_tokens=True) for response in response_tensors]
    return feedback_batch

def evaluate_actions_batch(scenes, goals, actions_batch):
    """Evaluate a batch of action sequences using the reward model."""
    with torch.no_grad():
        # Assuming reward_model can handle batched text inputs (list of scenes, goals, actions)
        reward_values = reward_model(scenes, goals, actions_batch)  # Returns a list or tensor of rewards
    return reward_values

def inference_batch(scenes, goals):
    """Inference loop to generate and refine action sequences for a batch."""
    batch_size = len(scenes)
    # Step 1: Initial action sequences from external LLM
    current_actions_batch = [llm_service.call(s, g, actions=None, feedback=None) for s, g in zip(scenes, goals)]
    iteration = 0
    completed = [False] * batch_size  # Track which examples are done
    final_actions_batch = [None] * batch_size  # Store final actions

    while iteration < MAX_ITERATIONS and not all(completed):
        # Step 2: Evaluate the current action sequences
        reward_values = evaluate_actions_batch(scenes, goals, current_actions_batch)
        print(f"Iteration {iteration + 1}: Rewards = {[f'{r}' for r in reward_values]}, Actions = {current_actions_batch}")

        # Step 3: Check which actions are good enough
        for i in range(batch_size):
            if not completed[i]:
                if reward_values[i] >= REWARD_THRESHOLD:
                    print(f"Example {i}: Action sequence accepted with reward {reward_values[i]}")
                    final_actions_batch[i] = current_actions_batch[i]
                    completed[i] = True

        # If all examples are completed, exit early
        if all(completed):
            break

        # Step 4: Generate feedback for incomplete examples
        active_scenes = [s for i, s in enumerate(scenes) if not completed[i]]
        active_goals = [g for i, g in enumerate(goals) if not completed[i]]
        active_actions = [a for i, a in enumerate(current_actions_batch) if not completed[i]]
        
        if active_actions:  # Only generate feedback if there are incomplete examples
            feedback_batch = generate_feedback_batch(active_scenes, active_goals, active_actions)
            print(f"Generated feedback for active examples: {feedback_batch}")

            # Step 5: Feed back to external LLM for incomplete examples
            new_actions_batch = [llm_service.call(s, g, a, f) for s, g, a, f in zip(active_scenes, active_goals, current_actions_batch, feedback_batch)]
            
            # Update current_actions_batch with new actions for incomplete examples
            active_idx = 0
            for i in range(batch_size):
                if not completed[i]:
                    current_actions_batch[i] = new_actions_batch[active_idx]
                    active_idx += 1

        iteration += 1

    # Handle any remaining incomplete examples
    for i in range(batch_size):
        if not completed[i]:
            reward = evaluate_actions_batch([scenes[i]], [goals[i]], [current_actions_batch[i]])[0]
            print(f"Example {i}: Max iterations ({MAX_ITERATIONS}) reached. Using best actions with reward {reward}")
            final_actions_batch[i] = current_actions_batch[i]

    return final_actions_batch

# Example usage
if __name__ == "__main__":
    sample_scenes = [
        "A robot is in a room with a locked door and a key on the table.",
        "A chef is in a kitchen with no ingredients.",
        "A driver is in a car with a flat tire."
    ]
    sample_goals = [
        "Unlock the door and exit the room.",
        "Prepare a meal for dinner.",
        "Drive to the nearest town."
    ]

    final_actions_batch = inference_batch(sample_scenes, sample_goals)
    print(f"Final action sequences: {final_actions_batch}")