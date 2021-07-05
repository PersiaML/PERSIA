pub mod hashmap_sharded_service;
pub mod middleware_config_parser;
pub mod monitor;
pub mod sharded_middleware_service;
pub mod sharding_utils;

pub mod persia_ndarray_serde {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    #[derive(Serialize, Deserialize)]
    struct Ndarray2Helper {
        shape_0: usize,
        shape_1: usize,
        content: serde_bytes::ByteBuf,
    }

    pub fn serialize<S>(
        array: &ndarray::Array2<f32>,
        serializer: S,
    ) -> Result<<S as Serializer>::Ok, <S as Serializer>::Error>
    where
        S: Serializer,
    {
        Ndarray2Helper {
            shape_0: array.shape()[0],
            shape_1: array.shape()[1],
            content: {
                let slice = array.as_slice().unwrap();
                let byte_slice = unsafe {
                    std::slice::from_raw_parts(
                        slice.as_ptr() as *const u8,
                        slice.len() * std::mem::size_of::<f32>(),
                    )
                };
                serde_bytes::ByteBuf::from(byte_slice)
            },
        }
        .serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<ndarray::Array2<f32>, D::Error>
    where
        D: Deserializer<'de>,
    {
        serde::Deserialize::deserialize(deserializer).map(
            |Ndarray2Helper {
                 shape_0,
                 shape_1,
                 content,
             }| {
                unsafe {
                    ndarray::Array2::from_shape_vec_unchecked((shape_0, shape_1), {
                        let slice = std::slice::from_raw_parts(
                            content.as_ptr() as *const f32,
                            content.len() / std::mem::size_of::<f32>(),
                        );
                        slice.to_vec()
                    })
                }
            },
        )
    }
}

pub mod persia_ndarray_serde_f16 {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    #[derive(Serialize, Deserialize)]
    struct Ndarray2Helper {
        shape_0: usize,
        shape_1: usize,
        content: serde_bytes::ByteBuf,
    }

    pub fn serialize<S>(
        array: &ndarray::Array2<half::f16>,
        serializer: S,
    ) -> Result<<S as Serializer>::Ok, <S as Serializer>::Error>
    where
        S: Serializer,
    {
        Ndarray2Helper {
            shape_0: array.shape()[0],
            shape_1: array.shape()[1],
            content: {
                let slice = array.as_slice().unwrap();
                let byte_slice = unsafe {
                    std::slice::from_raw_parts(
                        slice.as_ptr() as *const u8,
                        slice.len() * std::mem::size_of::<half::f16>(),
                    )
                };
                serde_bytes::ByteBuf::from(byte_slice)
            },
        }
        .serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<ndarray::Array2<half::f16>, D::Error>
    where
        D: Deserializer<'de>,
    {
        serde::Deserialize::deserialize(deserializer).map(
            |Ndarray2Helper {
                 shape_0,
                 shape_1,
                 content,
             }| {
                unsafe {
                    ndarray::Array2::from_shape_vec_unchecked((shape_0, shape_1), {
                        let slice = std::slice::from_raw_parts(
                            content.as_ptr() as *const half::f16,
                            content.len() / std::mem::size_of::<half::f16>(),
                        );
                        slice.to_vec()
                    })
                }
            },
        )
    }
}
