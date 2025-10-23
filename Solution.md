## Contents
- [Contents](#contents)
- [Code](#code)
- [Application Design](#application-design)
  - [Django/DRF](#djangodrf)
  - [Social Media Comments](#social-media-comments)
  - [Sentiment Analysis](#sentiment-analysis)
- [API Endpoints](#api-endpoints)
  - [Application Structure](#application-structure)
- [System Architecture](#system-architecture)
- [Security \& Authentication](#security--authentication)
- [Performance](#performance)
- [CI/CD](#cicd)
- [Maintenance](#maintenance)

## Code

Project: https://github.com/Sorath93/tool

SentimentAnalyzer: https://github.com/Sorath93/tool/tree/main/sentiment

## Application Design

### Django/DRF

I decided to use **Django** for the application and **Django REST Framework** to implement the API.
* Django's "batteries-included" philosophy allows for rapid development with minimal setup. Since this project is an MVP that needs to be put in front of users as soon as possible to validate value, the Django/DRF combination provides the best balance between speed and quality. Furthermore, the out-of-the-box combination of ORM for data models, authentication/authorization, and data validation means that we can focus on business logic.
* I looked into other frameworks like Flask and FastAPI, but chose not to use them for this exercise. 
  * Flask is lightweight, which would require setting up components from scratch (ORM, authentication, request handling).
  * FastAPI is a good modern option, especially for async use cases and OpenAPI specs, but, for me, it started to introduce extra learning overhead. I think it would be a good option if the application goes beyond MVP and use-case is validated! 
  
**Toolchain** 
- **Language**: Python 3.12.12
- **Frameworks**: Django & Django REST Framework
- **Task queue**: Celery (for job execution)
- **Broker**: Redis
- **Database**: SQLite in development. Can opt for PostgreSQL in production.

### Social Media Comments

**In terms of fetching comments** some considerations I would make are as follows:
* Do we want to store the comments? If so, we should probably implement behaviour that deletes comments that are more than X days/weeks old. For long term storage we could consider an object storage solution like Azure Blob Storage to store CSVs.
* Storing comments at this stage might not be necessary though - the project is described as "opportunistic" and with ROI inititally uncertain.
* Do the social media platforms have official APIs? What are the options?
  * It looks like the Instagram API is for Businesses/Creators to access their own data, TikTok requires a registration process, and Youtube has an official API that exposes comments (scraping Google Applications is against the Terms of Service).
  * From the above it seems as though we'll need to have a hybrid approach to begin with - scraping for Instagram and TikTok, and integrating Youtube API. 
  * Once ROI is proven we could consider third-party API providers (e.g. Apify).
* How do we make the process robust? E.g. failing to fetch comments or only being able to get a partial amount comments - we need to handle it - do we retry, move-on, fallback? Maybe we can have thresholds, for example, each post should have at least 50% of comments scraped? 
* Should we limit the number of comments scraped per post? How do we decide on a limit? Is there a sampling method we could use?
* What happens when there are no comments? What about if there's little amount of comments - how does this affect the overall sentiment score?
* We should probably implement logic that avoids fetching comments for a post that already has them in the database. E.g. if an Account Manager has two lists with common links.  

### Sentiment Analysis

**To implement the ``SentimentAnalyzer`` class**, I looked into three options: VADER, CardiffNLP Twitter RoBERTa (on Hugging Face), and Azure AI Language. I decided to go with CardiffNLP Twitter RoBERTa because it's trained on a huge amount of tweets (~124M), making it a reasonable choice for social media comments. It's also fine-tuned for sentiment analysis, handles hashtags and emojis, and can classify text as positive, neutral, or negative. It  also required low integration efforts - only needed two Python libraries to get going (``torch`` & ``transformers``) and the example pipeline on Hugging Face was enough to understand how to approach the code.

On the other hand, VADER is a rule-based lexicon method and weaker when it comes to context - for an application that is presented to stakeholders, even as opportunistic, I preferred the robustness of the Hugging Face model. Azure AI Language would have introduced unnecessary complexity for now (billing, API key storage, etc). However, I wanted to make the approach pluggable, therefore, implemented an Abstract Class and Factory Class. 

## API Endpoints

The URL pattern mapping can be seen in ``jobs/urls.py``

``api/jobs/create`` -> Accepts a list of URLs and executes the job.

``api/jobs/<int:job_id>/status`` -> GET the status of a job

``api/jobs/<int:job_id>/download/`` -> Download the results of a job as a CSV.

These URLs patterns map to the Django REST Framework views in ``jobs/views.py``. ``jobs/models.py`` defines the data models.

``jobs/tasks.py`` has the Celery tasks.

``tool/celery.py`` has the Celery app configuration. 

*Note that the API  and task code is not complete or tested - I just wanted to have a go!*

### Application Structure
```
.
├── jobs
│   ├── __init__.py
│   ├── admin.py
│   ├── apps.py
│   ├── migrations
│   │   ├── __init__.py
│   ├── models.py
│   ├── tasks.py
│   ├── tests.py
│   ├── urls.py
│   └── views.py
├── sentiment
│   ├── __init__.py
│   ├── analyzer.py
│   └── tests.py
└── tool
│   ├── __init__.py
│   ├── asgi.py
│   ├── celery.py
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
├── manage.py
├── requirements.txt
├── Solution.md
├── db.sqlite3
```

I decided to separate the Django project into two apps: ``jobs`` and ``sentiment``. The ``jobs`` app is treated as the orchestration layer (API handling and storage) and the ``sentiment`` app is where the sentiment analysis sits. 

My reasons for separation instead of one app:
* Clarity - it's obvious at a glance where to go when navigating the repo. Developer experience is important to me.
* Reusability and Scalability - You can extract the ``sentiment`` app as is. For example, if you decide on another framework, architecture pattern (e.g. microservices, different API protocol etc), or simply want to use it for another project or use-case.


## System Architecture

The information we have:
* No more than 10 users (but audience might grow in the future)
* Usage of the app is in bursts of activity rather than consistent
* Results do not need to be immediate and users will return to the application to retrieve results. 

This tells me that jobs need to be handled in the background and resources should scale dynamically based on demand. During quiet periods, the system should scale down to minimize costs but when bursts of activity occur it should automatically scale. 

This suggests an asynchronous task queue approach - the API creates a job, puts it in a queue, and a separate worker processes it in the background. The API immediately returns a response to the client so the user can continue using the application and check back later for the results. I chose Celery and Redis for this, which integrate easily with Django via the ``celery`` and ``redis`` Python libraries. Celery is for task execution and Redis is the message broker. 

Furthermore, Celery workers can scale horizontally when queue depth increases, i.e. when there are a lot of jobs waiting to be completed.  

![alt text](image-2.png)

In terms of **deploying** the application and **scalability** - there're two scenarios to consider:
* Increased number of users
* Increased number of jobs

This suggests that I could containerise the Django app and also containerise the Celery app to allow for independent scaling.

![alt text](image-3.png)

The diagram demonstrates the containers as Azure Container Apps inside an Azure Container Apps environment. Azure Container Apps are able to manage horizontal scaling through a set of declarative scaling rules (in our case increased requests and/or Redis queue depth). The documentation says "As a container app revision scales out, new instances (replicas) of the revision are created on-demand" - this suits our requirements of 'bursty' usage and possible increase in users.

Cost-efficiency is also a benefit of this setup - we can set the min/max replicas, meaning that we're not billed if a Container App scales to zero. This seems ideal for an opportunistic project. 

You'll also notice the Azure Blob Storage (for Object storage). This could possibly be used to store CSV files. 

## Security & Authentication

1. Users authenticate with Croud OIDC
2. Croud issues a JWT token
3. Client applications include the JWT token in API requests (`Authorization: Bearer <token>`)
4. Our Django API validates the JWT token
5. Authenticated users can access endpoints

![alt text](image-5.png)

The Django REST Framework documentation suggests using the Simple JWT (``djangorestframework-simplejwt``) library to do this. I would write a cusom authentication class to validate tokens against Croud's OIDC public key then add this authentication class to the API views via the ``@authentication_classes`` decorator.

**Secure Configuration Storage**

Sensitive information such as database credentials, Redis connection strings, API keys, etc. can be stored in Azure Key Vault. Django retrieves these values via `os.environ.get()` to avoid hard-coding. 

## Performance

## CI/CD 

I could use GitHub Actions for CI/CD. Whenever changes are merged into the main branch, a workflow will automatically run to:

1. run tests and linters,
2. build the Docker image,
3. push the image to a container registry (e.g. Azure Container Registry or Docker Hub), and trigger a deployment to Azure Container Apps using the latest image tag.

The Azure Container Apps services can be configured to pull the image from the registry.
// insert your own iimage
![alt text](image-4.png)

## Maintenance

## Sources

These are some sources I read

https://dataloop.ai/library/model/cardiffnlp_twitter-roberta-base-sentiment/#training-data-and-how-it-works 
https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest
https://medium.com/@jaxayprajapati/now-sentiment-analysis-becomes-effortless-with-azure-ai-language-f3e81089fb1b
https://github.com/public-apis/public-apis?tab=readme-ov-file#social
https://dsysd-dev.medium.com/system-design-patterns-producer-consumer-pattern-1572f813329b