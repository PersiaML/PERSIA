use syn::{Attribute, Ident, ReturnType, PatType, Receiver, FnArg, ImplItem};
use proc_macro::TokenStream;
use syn::export::TokenStream2;
use quote::ToTokens;

#[allow(dead_code)]
struct RpcMethod {
    attrs: Vec<Attribute>,
    ident: Ident,
    args: Vec<PatType>,
    receiver: Receiver,
    output: ReturnType,
}

impl RpcMethod {
    fn ident_web_api(&self) -> Ident {
        quote::format_ident!(
            "{}_web_api",
            self.ident
        )
    }

    fn req_type(&self) -> TokenStream2 {
        let args = &self.args;
        let all_arg_types: Vec<_> = args.iter().map(|x| &x.ty).collect();
        if all_arg_types.len() > 0 {
            quote::quote! {
                (#( #all_arg_types ),*,)
            }
        } else {
            quote::quote! {
                ()
            }
        }
    }

    fn resp_type(&self) -> TokenStream2 {
        let output = &self.output;
        let output = match output {
            ReturnType::Default => syn::Type::Verbatim(quote::quote! {()}),
            ReturnType::Type(_, t) => (**t).clone(),
        };
        quote::quote! {
            #output
        }
    }

    fn client_method(&self, client_field: &Ident) -> TokenStream2 {
        let ident = &self.ident;
        let ident_compressed = quote::format_ident!("{}_compressed", ident);
        let req_args: Vec<_> = self.args.iter().map(|x| &x.pat).collect();
        let original_args = self.args.clone();
        let resp_type = self.resp_type();
        let web_api_string_name = self.ident_web_api().to_string();

        if req_args.len() > 0 {
            quote::quote! {
                pub async fn #ident(&self, #( #original_args, )*) -> Result<#resp_type, ::persia_rpc::PersiaRpcError> {
                    let req = (#( #req_args, )*);
                    self.#client_field.call_async(#web_api_string_name, req, false).await
                }

                pub async fn #ident_compressed(&self, #( #original_args, )*) -> Result<#resp_type, ::persia_rpc::PersiaRpcError> {
                    let req = (#( #req_args, )*);
                    self.#client_field.call_async(#web_api_string_name, req, true).await
                }
            }
        } else {
            quote::quote! {
                pub async fn #ident(&self) -> Result<#resp_type, ::persia_rpc::PersiaRpcError> {
                    self.#client_field.call_async(#web_api_string_name, (), false).await
                }

                pub async fn #ident_compressed(&self) -> Result<#resp_type, ::persia_rpc::PersiaRpcError> {
                    self.#client_field.call_async(#web_api_string_name, (), true).await
                }
            }
        }
    }

    fn service_method(&self) -> TokenStream2 {
        let method_ident = &self.ident;
        let web_api_ident = self.ident_web_api();
        let web_api_ident_string = web_api_ident.to_string();
        let web_api_ident_compressed = quote::format_ident!("{}_compressed", web_api_ident);
        let web_api_ident_compressed_string = web_api_ident_compressed.to_string();
        let req_args: Vec<_> = self.args.iter().map(|x| &x.pat).collect();
        let req_args2 = req_args.clone();
        let req_type = self.req_type();
        let call_line  = if req_args.len() > 0 {
            quote::quote! {
                 let input: #req_type = ::tokio::task::block_in_place(|| ::bincode::deserialize(body.as_ref()))
                     .context(::persia_rpc::SerializationFailure {})?;
                 let (#( #req_args, )*) = input;
                 let output = self.#method_ident(#( #req_args2, )* ).await;
            }
        } else {
            quote::quote! {
                 let output = self.#method_ident().await;
            }
        };
        quote::quote! {
            pub async fn #web_api_ident(&self, req: ::hyper::Request<::hyper::Body>) -> Result<::hyper::Response<hyper::Body>, ::hyper::Error> {
                let result = async move {
                    let body =
                        ::hyper::body::to_bytes(req.into_body())
                            .await
                            .context(::persia_rpc::TransportError {
                                msg: format!("hyper read body error: {}", #web_api_ident_string),
                            })?;
                    #call_line
                    let output = ::tokio::task::block_in_place(|| ::bincode::serialize(&output))
                        .context(::persia_rpc::SerializationFailure {})?;
                    Ok::<_, ::persia_rpc::PersiaRpcError>(output)
                }
                .await;
                match result {
                    Ok(x) => Ok(::hyper::Response::new(::hyper::body::Body::from(x))),
                    Err(e) => {
                        ::tracing::error!("server side error {:?}", e);
                        let mut resp = ::hyper::Response::default();
                        *resp.status_mut() = ::hyper::StatusCode::INTERNAL_SERVER_ERROR;
                        *resp.body_mut() = ::hyper::body::Body::from(format!("{:#?}", e));
                        Ok(resp)
                    }
                }
            }


            pub async fn #web_api_ident_compressed(&self, req: ::hyper::Request<::hyper::Body>) -> Result<::hyper::Response<hyper::Body>, ::hyper::Error> {
                let result = async move {
                    let body =
                        ::hyper::body::to_bytes(req.into_body())
                            .await
                            .context(::persia_rpc::TransportError {
                                msg: format!("hyper read body error: {}", #web_api_ident_compressed_string),
                            })?;
                    let body = ::tokio::task::block_in_place(|| ::lz4::block::decompress(&body, None))
                        .context(::persia_rpc::IOFailure {})?;
                    #call_line
                    let output = ::tokio::task::block_in_place(|| ::bincode::serialize(&output))
                        .context(::persia_rpc::SerializationFailure {})?;
                    let output = ::tokio::task::block_in_place(|| ::lz4::block::compress(&output, None, true))
                        .context(::persia_rpc::IOFailure {})?;
                    Ok::<_, ::persia_rpc::PersiaRpcError>(output)
                }
                .await;
                match result {
                    Ok(x) => Ok(::hyper::Response::new(::hyper::body::Body::from(x))),
                    Err(e) => {
                        ::tracing::error!("server side error {:?}", e);
                        let mut resp = ::hyper::Response::default();
                        *resp.status_mut() = ::hyper::StatusCode::INTERNAL_SERVER_ERROR;
                        *resp.body_mut() = ::hyper::body::Body::from(format!("{:#?}", e));
                        Ok(resp)
                    }
                }
            }
        }
    }
}

struct Service {
    rpcs: Vec<RpcMethod>,
    service_type_name: String,
}

impl Service {
    fn client_impl(&self) -> TokenStream2 {
        let client_type_name = quote::format_ident!("{}Client", self.service_type_name);
        let client_field = quote::format_ident!("inner_rpc_client");
        let client_methods: Vec<_> = self.rpcs.iter().map(|x| x.client_method(&client_field)).collect();
        quote::quote! {
            pub struct #client_type_name {
                #client_field: ::persia_rpc::RpcClient
            }

            impl #client_type_name {
                pub fn new(client: ::persia_rpc::RpcClient) -> Self {
                    Self {
                        #client_field: client
                    }
                }

                #( #client_methods )*
            }
        }
    }

    fn service_impl(&self) -> TokenStream2 {
        let service_ident = quote::format_ident!("{}", self.service_type_name);
        let service_web_api_impl: Vec<_> = self.rpcs.iter().map(|x| x.service_method()).collect();
        let match_arms: Vec<_> = self.rpcs.iter().map(|x| {
            let web_api_string_name = "/".to_string() + x.ident_web_api().to_string().as_str();
            let ident_web_api = x.ident_web_api();
            quote::quote! {
                #web_api_string_name => {
                    Box::pin(async move { server.#ident_web_api(req).await })
                }
            }
        }).collect();
        quote::quote! {
            use std::task::{Context, Poll};

            impl #service_ident {
                #( #service_web_api_impl )*
            }

            impl ::hyper::service::Service<::hyper::Request<::hyper::Body>> for #service_ident {
                type Response = ::hyper::Response<::hyper::Body>;
                type Error = ::hyper::Error;
                type Future = std::pin::Pin<Box<dyn std::future::Future<Output = Result<Self::Response, Self::Error>> + Send>>;

                fn poll_ready(&mut self, _ctx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
                    Poll::Ready(Ok(()))
                }

                fn call(&mut self, req: ::hyper::Request<::hyper::Body>) -> Self::Future {
                    let server = self.clone();
                    match req.uri().path() {
                        #( #match_arms )*
                        _ => {
                            Box::pin(
                                async move {
                                    let mut not_found = hyper::Response::default();
                                    *not_found.status_mut() = hyper::StatusCode::NOT_FOUND;
                                    Ok(not_found)
                                }
                            )
                        }
                    }
                }
            }
        }
    }
}

#[proc_macro_attribute]
pub fn service(_attr: TokenStream, tokens: TokenStream) -> TokenStream {
    let item = syn::parse_macro_input!(tokens as syn::ItemImpl);
    let service_name = item.self_ty.clone().into_token_stream().to_string();
    let rpc_methods: Vec<_> = item.items.iter()
        .filter_map(|item| match item {
            ImplItem::Method(m) => Some(m),
            _ => None,
        })
        .map(|m| {
            let mut args = vec![];
            let mut receiver = None;
            for arg in &m.sig.inputs {
                match arg {
                    FnArg::Receiver(captures) => {
                        if captures.mutability.is_some() {
                            panic!("should be &self");
                        } else if receiver.is_some() {
                            panic!("duplicated self");
                        } else {
                            receiver = Some(captures.clone());
                        }
                    }
                    FnArg::Typed(captures) => match *captures.pat {
                        syn::Pat::Ident(_) => args.push(captures.clone()),
                        _ => panic!("patterns aren't allowd in RPC args"),
                    },
                }
            }
            RpcMethod {
                attrs: m.attrs.clone(),
                ident: m.sig.ident.clone(),
                args,
                receiver: receiver.unwrap(),
                output: m.sig.output.clone(),
            }
        })
        .collect();

    let s = Service {
        rpcs: rpc_methods,
        service_type_name: service_name,
    };

    let client_impl = s.client_impl();
    let service_impl = s.service_impl();

    (quote::quote! {
        #item

        #service_impl

        #client_impl
    }).into()
}
