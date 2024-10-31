import gradio as gr
import requests


def generate(input_str, init_prompts):
    r = requests.post(
        "http://10.0.0.10/generate",
        json={"input_str": input_str, "soft_prompts": init_prompts},
    )

    return f"{r.text}"


demo = gr.Interface(
    fn=generate,
    inputs=[
        "text",
        gr.CheckboxGroup(
            ["SlovakAlpaca", "SST2"],
            label="Soft Prompt",
            info="How to set the soft prompt?",
        ),
    ],
    outputs=["text"],
)

if __name__ == "__main__":
    demo.launch()
