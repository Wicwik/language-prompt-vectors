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
        gr.Textbox(label="Input"),
        gr.CheckboxGroup(
            ["SlovakAlpaca", "SST2"],
            label="Init prompt",
            info="How to set the soft prompt? (choosing multiple is sum of TPV)",
        ),
    ],
    outputs=[gr.Textbox(label="Output")],
    examples=[
        "Vytvorte tabuľku SQL s nasledujúcimi stĺpcami: _id, name, phone, email.",
        ["SlovakAlpaca"],
    ],
)

if __name__ == "__main__":
    demo.launch()
