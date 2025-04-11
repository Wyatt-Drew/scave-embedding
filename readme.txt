The backend is set up on Render.  It relies on some environment variables there.
- Importantly it is on a Free Tier.  This means that it shuts down when not being used.
That means that the first query you do after a while may fail.  We set up UpTimeRobot to ping it
every 5 minutes, but sometimes it still shuts down.

In general it can be said that the server is running things and there is
a number of routes that call the database.  We have frozen the database for submission, 
so this will be out of date by the time that this is submitted.

Here are example API calls:
https://scave-embedding.onrender.com/api/products/SemanticSearch?query=potato
- Completes semantic SemanticSearch

https://scave-embedding.onrender.com/api/prices/GetProductHistory?product_num=20000005
- Get historical data for a particular product across many stores

https://scave-embedding.onrender.com/api/products/GetProduct?search=roma
- Keyword search

https://scave-embedding.onrender.com/api/products/GetDeals
- Returns deals for this week.