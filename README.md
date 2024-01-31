# jobs



## AI package
This package has three components:
 - job_postings_pipeline.py
 - application_questions_pipeline_b.py
 - resume_pipeline.py

 

 ### Job postings pipeline
 This AI pipeline deals with job postings. It takes scraped job postings, extracts key hard skills, vectorizes them to embeddings and creates a searchable kdtree. This allows us to query the kdtree for a closest n jp by feeding in an embedded hard skills from a resume.

  ### Application question pipeline
 This AI pipeline deals with job application questions. It takes scraped job application questions, and tries to generalize them, compressing similar questions into one.
 Given a new application question, it also allows us to find n number of similar historical qa-pairs and returns them, Then it can take these and tune them via chatgpt to produce m different answers. Both n and m are definable in `.env`

  ### Resume pipeline
 This AI pipeline deals with resumes. It takes scraped resumes, extracts key hard skills, vectorizes them to embeddings. These are used to query the kdtree created in `Job postings pipeline`

 

 Check out the miro board for a visual depiction of these modules:
 https://miro.com/app/board/uXjVMJMRyjg=/




## Project installation

```
cd existing_repo
git remote add origin https://src.uacaus.com/development/ai/jobs.git
git branch -M main
git push -uf origin main
```

copy over the latest folder structure + artifacts from google drive

Set up projects for absolute imports
1. navigate to `AI_CORE/` folder
2. install package as editable via pip
`pip install -e .`
3. install poetry
`pip install poetry`

4. Install requirements
`poetry install`

## Contributing
Please create branches from `develop`, with this naming structure:
`feature/your-branch-name`
Merge back to `develop`. Ask `stepan.l` or `marianna.p` and `pavlo` to review your code.


## Authors and acknowledgment
Repository authors:

Stepan Lavrinenko
Marianna Petrova

