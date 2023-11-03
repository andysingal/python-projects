# python-projects

Content from https://towardsdatascience.com/when-automl-meets-large-language-model-756e6bb9baa7

While approaching the hyperparameter tuning problems from a pure algorithmic point of view can work, there is another crucial piece of information that could complement the algorithmic search strategy: human expertise.

Senior data scientists, based on their years of experience and deep understanding of the ML algorithms, often intuitively know where to initiate the search, which regions of the search space might be more promising, or when to narrow down or expand the possibilities.

Therefore, this gives us a very interesting idea: can we design a scalable expertise-guided search strategy that leverages both the nuanced insights provided by the expert and the search efficiency offered by the AutoML algorithms?

This is where the large language models (LLM), such as GPT-4, can play a role. Among their vast amount of training data, a certain portion of the corpus contains texts that are dedicated to explaining and discussing the best practices of machine learning (ML). Thanks to that, the LLMs are able to internalize a substantial amount of ML expertise and obtain a significant chunk of the collective ML wisdom. This positions LLMs as potential knowledgable ML experts who can interact with the existing AutoML tool to collaboratively perform expert-guided hyperparameter tuning.

<img width="482" alt="Screenshot 2023-11-02 at 1 47 18 PM" src="https://github.com/andysingal/python-projects/assets/20493493/4a649cc9-cdae-43f6-b825-52fbfa8a6971">




Resources:
- https://mlops.community/blog/
- https://geekflare.com/best-mlops-platforms/
