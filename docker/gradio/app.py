import gradio as gr


def generate(input_str, init_prompts):
    return f"{input_str}, {init_prompts}"


demo = gr.Interface(
    fn=generate,
    inputs=["text", 
            gr.CheckboxGroup(["SST2", "SlovakAlpaca"], label="Soft Prompt", info="How to set the soft prompt?"),],
    outputs=["text"],
)

if __name__ == "__main__":
    demo.launch()
