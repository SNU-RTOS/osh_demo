FROM golang:1.24 AS build
WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download
COPY api api
COPY controller controller
COPY cmd cmd
RUN CGO_ENABLED=0 go build -trimpath -o /controller ./cmd/controller
FROM gcr.io/distroless/static:nonroot
COPY --from=build /controller /controller
USER 65532:65532
ENTRYPOINT ["/controller"]
