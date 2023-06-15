FROM node:current

ARG BACKEND_SERVER

RUN git clone https://github.com/thesstefan/teshub
WORKDIR teshub/website

RUN yarn install

RUN echo "$BACKEND_SERVER"

RUN sed -i "s/127.0.0.1/$BACKEND_SERVER/g" package.json

CMD ["yarn", "start"]
