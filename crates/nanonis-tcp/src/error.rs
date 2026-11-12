#[derive(Debug, thiserror::Error)]
pub enum NanonisTcpError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("nanonis api error: {0}")]
    Api(String),
    #[error("codec error: {0}")]
    Codec(#[from] CodecError),
}

pub type NanonisTcpResult<T> = std::result::Result<T, NanonisTcpError>;

#[derive(Debug, thiserror::Error)]
pub enum CodecError {
    #[error("wrong command name received: expected `{expected}` but got `{received}`")]
    NameMismatch { expected: String, received: String },
    #[error("did not parse all received bytes: expected `{expected}` but parsed `{parsed}`")]
    ReadLenMismatch { expected: usize, parsed: usize },
    #[error("did not write full buffer: expected `{expected}` but wrote `{wrote}`")]
    WriteLenMismatch { expected: usize, wrote: usize },
    #[error("tried to create a fixed string of `{string}` that exceed the max length `{maximum}`")]
    StringTooLong { maximum: usize, string: String },
}
